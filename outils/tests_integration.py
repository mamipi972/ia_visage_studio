#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Execution complete du greffon hors de GIMP.

Usage : python3 outils/tests_integration.py   (necessite numpy et opencv)

Les autres bancs couvrent les morceaux : la doublure d'API exerce les
fonctions du greffon une par une, la doublure d'onnxruntime exerce le worker.
Il manquait celui-ci : run_procedure du debut a la fin, avec un vrai
sous-processus, un vrai fichier de parametres et un vrai fichier de sortie.

Un greffon qui s'ouvre n'est pas un greffon qui fonctionne : la validation
porte donc sur le calque produit, son nom, sa place dans la pile, et sur le
code de retour rendu a GIMP - jamais sur l'absence de message d'erreur.
"""

import json
import os
import shutil
import sys
import tempfile

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import faux_gimp
import tests_worker

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHEMIN_GREFFON = os.path.join(RACINE, "ia_visage_studio.py")

RESULTATS = []


def controler(nom, condition, detail=""):
    RESULTATS.append((nom, bool(condition), detail))
    marque = "OK   " if condition else "ECHEC"
    suffixe = (" - " + str(detail)) if (detail and not condition) else ""
    print("  %s %s%s" % (marque, nom, suffixe))
    return bool(condition)


# ---------------------------------------------------------------------------
# Doublures d'objets GIMP qui portent reellement des pixels.
# ---------------------------------------------------------------------------
class FauxCalque:
    def __init__(self, nom, tableau, parent=None):
        self.nom = nom
        self.tableau = tableau
        self.parent = parent
        self.offsets = (0, 0)
        self.visible = True

    def get_name(self):
        return self.nom

    def set_name(self, nom):
        self.nom = nom

    def get_parent(self):
        return self.parent

    def get_offsets(self):
        # Un booleen precede volontairement les valeurs utiles : c'est la
        # disposition qu'aucune lecture par indice ne doit supposer.
        return (True, self.offsets[0], self.offsets[1])

    def set_offsets(self, x, y):
        self.offsets = (int(x), int(y))

    def set_visible(self, valeur):
        self.visible = bool(valeur)

    def set_opacity(self, valeur):
        return None

    def set_mode(self, mode):
        return None

    def has_alpha(self):
        return self.tableau.ndim == 3 and self.tableau.shape[2] == 4

    def add_alpha(self):
        return None


class FauxImage:
    def __init__(self, tableau):
        self.tableau = tableau
        self.calques = []
        self.undo_ouverts = 0
        self.supprimee = False

    def get_width(self):
        return self.tableau.shape[1]

    def get_height(self):
        return self.tableau.shape[0]

    @staticmethod
    def get_base_type():
        return 0

    @staticmethod
    def get_precision():
        return 0

    def insert_layer(self, calque, parent, position):
        calque.parent = parent
        self.calques.insert(max(0, int(position)), calque)

    def get_item_position(self, item):
        return self.calques.index(item)

    def undo_group_start(self):
        self.undo_ouverts += 1

    def undo_group_end(self):
        self.undo_ouverts -= 1

    def delete(self):
        self.supprimee = True


class FauxFichier:
    def __init__(self, chemin):
        self.chemin = chemin


class FauxGio:
    class File:
        @staticmethod
        def new_for_path(chemin):
            return FauxFichier(chemin)


class FauxProcedure:
    def __init__(self):
        self.retours = []

    def new_return_values(self, statut, erreur):
        self.retours.append((statut, erreur))
        return (statut, erreur)


class Config:
    def __init__(self, valeurs):
        self.valeurs = valeurs

    def get_property(self, nom):
        if nom not in self.valeurs:
            raise AttributeError(nom)
        return self.valeurs[nom]


def brancher_pixels(module):
    """Rend file_save et file_load_layer reellement fonctionnels."""

    def image_neuve(largeur, hauteur, *reste):
        return FauxImage(np.zeros((hauteur, largeur, 3), dtype=np.uint8))

    def calque_depuis(drawable, image_cible):
        return FauxCalque(drawable.get_name(), drawable.tableau)

    def file_save(run_mode, image, drawables, fichier):
        if isinstance(drawables, (list, tuple)):
            drawable = drawables[0]
        else:
            drawable = drawables
        if not isinstance(fichier, FauxFichier):
            raise TypeError("variante d'API non prise en charge par la doublure")
        if not cv2.imwrite(fichier.chemin, drawable.tableau):
            raise RuntimeError("ecriture refusee")
        return True

    def file_load_layer(run_mode, image, fichier):
        tableau = cv2.imread(fichier.chemin, cv2.IMREAD_UNCHANGED)
        if tableau is None:
            raise RuntimeError("lecture impossible")
        return FauxCalque("resultat", tableau)

    faux_gimp.FauxGimp.Image = type("Image", (), {
        "new_with_precision": staticmethod(image_neuve),
        "new": staticmethod(image_neuve)})
    faux_gimp.FauxGimp.Layer = type("Layer", (), {
        "new_from_drawable": staticmethod(calque_depuis)})
    faux_gimp.FauxGimp.file_save = staticmethod(file_save)
    faux_gimp.FauxGimp.file_load_layer = staticmethod(file_load_layer)
    module.Gio = FauxGio
    module.Gimp.Image = faux_gimp.FauxGimp.Image
    module.Gimp.Layer = faux_gimp.FauxGimp.Layer


class Bac:
    """Profil GIMP, dossier de donnees, modeles factices et faux venv."""

    def __init__(self, visages, operations_env=None):
        self.dossier = tempfile.mkdtemp(prefix="test_integration_")
        self.profil = os.path.join(self.dossier, "profil_gimp")
        self.donnees = os.path.join(self.dossier, "donnees")
        self.faux_paquets = os.path.join(self.dossier, "faux_paquets")
        for chemin in (self.profil, self.donnees, self.faux_paquets):
            os.makedirs(chemin, exist_ok=True)
        with open(os.path.join(self.faux_paquets, "onnxruntime.py"), "w") as f:
            f.write(tests_worker.FAUX_ONNXRUNTIME)

        self.env_sauve = {}
        reglages = {
            "LOCALAPPDATA": self.donnees, "XDG_DATA_HOME": self.donnees,
            "HOME": self.donnees, "GIMP_AI_SUITE_DIR": None, "CUDA_PATH": None,
            "TEST_TRACE": os.path.join(self.dossier, "trace.jsonl"),
            "TEST_VISAGES": json.dumps(visages),
            "TEST_SOURCE_GRISE": "0",
        }
        reglages.update(operations_env or {})
        for cle, valeur in reglages.items():
            self.env_sauve[cle] = os.environ.get(cle)
            if valeur is None:
                os.environ.pop(cle, None)
            else:
                os.environ[cle] = valeur

        faux_gimp.retirer_sonde_annulation()
        self.module = faux_gimp.installer(self.profil, CHEMIN_GREFFON)
        brancher_pixels(self.module)

        # Faux venv : l'interpreteur courant fait l'affaire, et le marqueur
        # evite toute installation. C'est exactement l'etat d'un poste deja
        # installe - l'etat que les tests oublient le plus souvent de couvrir.
        self.module.ecrire_marqueur({"statut": "pret",
                                     "venv_python": sys.executable,
                                     "canal": "test", "decouvert_le": "2026-01-01"})

        # Le worker tourne avec clean_env(), qui purge PYTHONPATH : on le
        # reinjecte, faute de quoi il ne trouverait pas la doublure.
        original = self.module.clean_env

        def clean_env(dossiers_dll=None):
            env = original(dossiers_dll)
            env["PYTHONPATH"] = self.faux_paquets
            return env

        self.module.clean_env = clean_env

    def poser_modeles(self, roles):
        for role in roles:
            nom = self.module.MODELES[role]["fichier"]
            chemin = os.path.join(self.module.get_models_dir(), nom)
            with open(chemin, "wb") as f:
                f.write(b"\x08\x07onnx factice" + b"\x00" * (2 * 1024 * 1024))

    def fermer(self):
        for cle, valeur in self.env_sauve.items():
            if valeur is None:
                os.environ.pop(cle, None)
            else:
                os.environ[cle] = valeur
        faux_gimp.retirer_sonde_annulation()
        shutil.rmtree(self.dossier, ignore_errors=True)


def image_source(largeur, hauteur):
    return tests_worker.image_de_test(largeur, hauteur, np.uint8, 3)


OPTIONS_PAR_DEFAUT = {
    "anonymize": False, "smile": False, "enhance": False, "colorize": False,
    "anonymize-mode": 0, "max-faces": 32, "smile-intensity": 1.0,
    "allow-download": False, "use-gpu": False, "reinstall-env": False,
}


def executer(bac, options, largeur=640, hauteur=480):
    module = bac.module
    tableau = image_source(largeur, hauteur)
    image = FauxImage(tableau)
    calque = FauxCalque("Portrait", tableau)
    image.calques.append(calque)
    proc = FauxProcedure()
    valeurs = dict(OPTIONS_PAR_DEFAUT)
    valeurs.update(options)
    greffon = module.IaVisageStudioPlugin()
    statut, erreur = greffon.run_procedure(
        proc, module.Gimp.RunMode.NONINTERACTIVE, image, [calque],
        Config(valeurs), None)
    return {"statut": statut, "erreur": erreur, "image": image,
            "messages": list(faux_gimp.MESSAGES)}


def test_anonymisation_de_bout_en_bout():
    print("Integration : anonymisation complete, du filtre au calque")
    visages = [[200.0, 140.0, 120.0, 140.0, 0.93], [450.0, 250.0, 90.0, 100.0, 0.72]]
    bac = Bac(visages)
    try:
        bac.poser_modeles([bac.module.ROLE_DETECTION])
        sortie = executer(bac, {"anonymize": True})
        module = bac.module
        controler("le filtre rend SUCCESS",
                  sortie["statut"] == module.Gimp.PDBStatusType.SUCCESS,
                  "%s | %s" % (sortie["statut"], sortie["messages"]))
        calques = sortie["image"].calques
        controler("un calque a ete ajoute au-dessus de l'original",
                  len(calques) == 2 and calques[0].nom != "Portrait",
                  str([c.nom for c in calques]))
        if len(calques) < 2:
            return
        nom = calques[0].nom
        controler("le nom du calque nomme le moteur reellement employe",
                  "yolov8n-face.onnx" in nom and "anonymisation" in nom, nom)
        controler("et le materiel de calcul", "processeur" in nom, nom)
        controler("le calque produit porte les dimensions de l'image",
                  calques[0].tableau.shape[:2] == (480, 640),
                  str(calques[0].tableau.shape))
        controler("le groupe d'annulation est referme",
                  sortie["image"].undo_ouverts == 0,
                  str(sortie["image"].undo_ouverts))

        inventaire = json.load(open(module.get_inventory_path(), encoding="utf-8"))
        controler("l'inventaire enregistre l'execution",
                  inventaire.get("visages_traites") == 2
                  and inventaire.get("greffon") == module.PLUGIN_ID,
                  str(inventaire.get("visages_traites")))
        controler("l'inventaire nomme le dossier de donnees et son contenu",
                  inventaire.get("dossier_donnees")
                  and inventaire.get("modeles"), str(inventaire)[:200])
    finally:
        bac.fermer()


def test_preseance_de_bout_en_bout():
    print("Integration : anonymisation + amelioration, arbitrage du greffon")
    visages = [[200.0, 140.0, 120.0, 140.0, 0.93]]
    bac = Bac(visages)
    try:
        module = bac.module
        bac.poser_modeles([module.ROLE_DETECTION, module.ROLE_AMELIORATION])
        sortie = executer(bac, {"anonymize": True, "enhance": True})
        controler("le filtre rend SUCCESS",
                  sortie["statut"] == module.Gimp.PDBStatusType.SUCCESS,
                  "%s | %s" % (sortie["statut"], sortie["messages"]))
        controler("l'utilisateur est prevenu que l'amelioration est ignoree",
                  any("prioritaire" in m for m in sortie["messages"]),
                  str(sortie["messages"]))

        trace = os.path.join(bac.dossier, "trace.jsonl")
        roles = []
        if os.path.isfile(trace):
            for ligne in open(trace, encoding="utf-8"):
                ligne = ligne.strip()
                if ligne:
                    evenement = json.loads(ligne)
                    if evenement["evenement"] == "session":
                        roles.append(evenement["role"])
        controler("aucun modele d'amelioration n'a ete charge",
                  roles == ["detection"], str(roles))
        controler("aucune anomalie d'arbitrage n'est signalee",
                  not any("Anomalie interne" in m for m in sortie["messages"]),
                  str(sortie["messages"]))
    finally:
        bac.fermer()


def test_degradation_de_bout_en_bout():
    print("Integration : modeles absents, telechargement decoche")
    visages = [[200.0, 140.0, 120.0, 140.0, 0.93]]
    bac = Bac(visages)
    try:
        module = bac.module
        bac.poser_modeles([module.ROLE_DETECTION])
        sortie = executer(bac, {"enhance": True, "smile": True})
        controler("le filtre rend tout de meme SUCCESS",
                  sortie["statut"] == module.Gimp.PDBStatusType.SUCCESS,
                  "%s | %s" % (sortie["statut"], sortie["messages"]))
        calques = sortie["image"].calques
        controler("un calque est produit malgre les modeles manquants",
                  len(calques) == 2, str([c.nom for c in calques]))
        messages = " ".join(sortie["messages"])
        controler("le sourire ignore est explique", "sourire" in messages,
                  messages[:300])
        # Deux mecanismes disent la meme chose : la resolution des modeles,
        # cote greffon, et la degradation constatee par le worker. Seule cette
        # formulation-la vient du worker, et c'est donc la seule qui teste le
        # chemin worker -> utilisateur.
        controler("la degradation constatee par le worker remonte telle quelle",
                  "aucun modele utilisable" in messages, messages[:300])
        controler("l'amelioration annonce son repli sans IA",
                  "sans IA" in messages, messages[:300])
        if len(calques) >= 2:
            controler("le nom du calque annonce le repli",
                      "sans IA" in calques[0].nom, calques[0].nom)
    finally:
        bac.fermer()


def test_aucune_operation():
    print("Integration : aucune case cochee")
    bac = Bac([])
    try:
        module = bac.module
        sortie = executer(bac, {})
        controler("le filtre rend CANCEL plutot qu'une erreur",
                  sortie["statut"] == module.Gimp.PDBStatusType.CANCEL,
                  str(sortie["statut"]))
        controler("et dit quoi faire",
                  any("Cochez au moins" in m for m in sortie["messages"]),
                  str(sortie["messages"]))
    finally:
        bac.fermer()


def test_echec_archive_les_journaux():
    print("Integration : echec de traitement, journaux archives")
    bac = Bac([])
    try:
        module = bac.module
        bac.poser_modeles([module.ROLE_DETECTION])
        # Aucun visage a trouver, et aucune autre operation : le worker doit
        # echouer, et l'echec doit laisser une trace exploitable.
        sortie = executer(bac, {"anonymize": True})
        controler("le filtre rend EXECUTION_ERROR",
                  sortie["statut"] == module.Gimp.PDBStatusType.EXECUTION_ERROR,
                  str(sortie["statut"]))
        messages = " ".join(sortie["messages"])
        controler("le message explique la cause en francais",
                  "visage" in messages, messages[:300])
        controler("le message cite le dossier d'archivage",
                  "Journaux archives" in messages, messages[:400])
        archives = []
        racine = module.get_logs_dir()
        if os.path.isdir(racine):
            archives = [d for d in os.listdir(racine)
                        if os.path.isdir(os.path.join(racine, d))]
        controler("les journaux sont bien sur le disque", len(archives) == 1,
                  str(archives))
        if archives:
            contenu = os.listdir(os.path.join(racine, archives[0]))
            controler("l'archive contient les parametres et le journal du worker",
                      "cfg.json" in contenu and "worker.log" in contenu,
                      str(contenu))
    finally:
        bac.fermer()


def test_dossier_temporaire_detruit():
    print("Integration : le dossier d'execution ne survit pas au traitement")
    visages = [[200.0, 140.0, 120.0, 140.0, 0.93]]
    bac = Bac(visages)
    try:
        module = bac.module
        bac.poser_modeles([module.ROLE_DETECTION])
        avant = set(os.listdir(tempfile.gettempdir()))
        executer(bac, {"anonymize": True})
        apres = set(os.listdir(tempfile.gettempdir()))
        restes = [d for d in (apres - avant) if d.startswith(module.PLUGIN_ID)]
        controler("aucun dossier de travail ne reste dans le temp global",
                  not restes, str(restes))
    finally:
        bac.fermer()


def test_enregistrement_protege():
    """Une exception sur un seul appel d'enregistrement ferait disparaitre le
    greffon en entier, des menus comme du navigateur de procedures, sans
    aucun message."""
    print("Integration : chaque appel d'enregistrement est protege")
    bac = Bac([])
    try:
        module = bac.module
        appels = []

        class ProcedureCapricieuse:
            def __getattr__(self, nom):
                def appel(*args, **kwargs):
                    appels.append(nom)
                    if nom == "set_menu_label":
                        raise RuntimeError("cette build n'aime pas les menus")
                    return None
                return appel

        capricieuse = ProcedureCapricieuse()
        module.Gimp.ImageProcedure = type("ImageProcedure", (), {
            "new": staticmethod(lambda *a, **k: capricieuse)})
        greffon = module.IaVisageStudioPlugin()
        rendu = greffon.do_create_procedure("plug-in-ia-visage-studio")
        controler("l'enregistrement aboutit malgre un appel en echec",
                  rendu is capricieuse)
        controler("les enregistrements suivants ont tout de meme eu lieu",
                  "add_menu_path" in appels and "set_documentation" in appels,
                  str(appels))
        controler("les sept cases et les trois reglages sont declares",
                  appels.count("add_boolean_argument") == 7
                  and appels.count("add_int_argument") == 2
                  and appels.count("add_double_argument") == 1,
                  str([a for a in appels if "argument" in a]))

        journal = os.path.join(module.get_shared_dir(),
                               "journal_" + module.PLUGIN_ID + ".log")
        controler("l'echec d'enregistrement laisse une trace dans le journal",
                  os.path.isfile(journal)
                  and "set_menu_label" in open(journal, encoding="utf-8").read(),
                  journal)
    finally:
        bac.fermer()


def main():
    print("Tests d'integration : run_procedure de bout en bout, hors de GIMP")
    print("")
    test_anonymisation_de_bout_en_bout()
    test_preseance_de_bout_en_bout()
    test_degradation_de_bout_en_bout()
    test_aucune_operation()
    test_echec_archive_les_journaux()
    test_dossier_temporaire_detruit()
    test_enregistrement_protege()
    print("")
    echecs = [nom for nom, ok, _ in RESULTATS if not ok]
    print("%d controles, %d echec(s)" % (len(RESULTATS), len(echecs)))
    for nom in echecs:
        print("  ECHEC : " + nom)
    return 1 if echecs else 0


if __name__ == "__main__":
    sys.exit(main())
