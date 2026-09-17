#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Tests du greffon hors de GIMP, par la doublure de l'API.

Usage : python3 outils/tests_unitaires.py   (aucune dependance externe)

Ce que ces tests couvrent, et que rien d'autre ne couvre : l'arbitrage des
intentions contradictoires, la decouverte de l'interpreteur sur une sortie
reelle capturee, les emplacements de fichiers, le marqueur d'environnement, la
resolution des modeles face a un vrai serveur HTTP local, le refus de l'option
GPU avant tout telechargement, l'archivage des journaux, et la garantie qu'un
processus enfant ne survit pas a son parent.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import faux_gimp

CHEMIN_GREFFON = os.path.join(RACINE, "ia_visage_studio.py")
SORTIES_REELLES = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "sorties_reelles")

RESULTATS = []


def controler(nom, condition, detail=""):
    RESULTATS.append((nom, bool(condition), detail))
    marque = "OK   " if condition else "ECHEC"
    suffixe = (" - " + str(detail)) if (detail and not condition) else ""
    print("  %s %s%s" % (marque, nom, suffixe))
    return bool(condition)


class Bac:
    """Faux profil GIMP et faux dossier de donnees, isoles du poste."""

    def __init__(self):
        self.dossier = tempfile.mkdtemp(prefix="test_visage_studio_")
        self.profil = os.path.join(self.dossier, "profil_gimp")
        self.donnees = os.path.join(self.dossier, "donnees")
        os.makedirs(self.profil, exist_ok=True)
        os.makedirs(self.donnees, exist_ok=True)
        self.env_sauve = {}
        for cle, valeur in (("LOCALAPPDATA", self.donnees),
                            ("XDG_DATA_HOME", self.donnees),
                            ("HOME", self.donnees),
                            ("GIMP_AI_SUITE_DIR", None),
                            ("CUDA_PATH", None)):
            self.env_sauve[cle] = os.environ.get(cle)
            if valeur is None:
                os.environ.pop(cle, None)
            else:
                os.environ[cle] = valeur
        faux_gimp.retirer_sonde_annulation()
        self.module = faux_gimp.installer(self.profil, CHEMIN_GREFFON)

    def modeles(self):
        return self.module.get_models_dir()

    def fermer(self):
        for cle, valeur in self.env_sauve.items():
            if valeur is None:
                os.environ.pop(cle, None)
            else:
                os.environ[cle] = valeur
        faux_gimp.retirer_sonde_annulation()
        shutil.rmtree(self.dossier, ignore_errors=True)


class ServeurModeles:
    """Sert des fichiers factices et compte les requetes, par methode."""

    def __init__(self, fichiers, tailles_annoncees=None):
        self.fichiers = fichiers
        # Taille annoncee par HEAD quand elle differe de celle servie : c'est
        # ainsi qu'un modele trop lourd se teste sans dependre d'une adresse
        # morte, qui ferait echouer le telechargement pour une autre raison et
        # rendrait le test indifferent au seuil.
        self.tailles_annoncees = tailles_annoncees or {}
        self.requetes = []
        parent = self

        class Gestionnaire(BaseHTTPRequestHandler):
            def log_message(self, *args):
                return

            def _contenu(self):
                return parent.fichiers.get(self.path.lstrip("/"))

            def do_HEAD(self):
                parent.requetes.append(("HEAD", self.path))
                contenu = self._contenu()
                if contenu is None:
                    self.send_response(404)
                    self.end_headers()
                    return
                nom = self.path.lstrip("/")
                taille = parent.tailles_annoncees.get(nom, len(contenu))
                self.send_response(200)
                self.send_header("Content-Length", str(taille))
                self.end_headers()

            def do_GET(self):
                parent.requetes.append(("GET", self.path))
                contenu = self._contenu()
                if contenu is None:
                    self.send_response(404)
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header("Content-Length", str(len(contenu)))
                self.end_headers()
                self.wfile.write(contenu)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Gestionnaire)
        self.fil = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.fil.start()

    @property
    def base(self):
        hote, port = self.httpd.server_address[:2]
        return "http://%s:%d/" % (hote, port)

    def gets(self):
        return [r for r in self.requetes if r[0] == "GET"]

    def fermer(self):
        self.httpd.shutdown()
        self.httpd.server_close()


def faux_modele(octets):
    """Contenu qui passe le controle de vraisemblance : protobuf plausible."""
    return b"\x08\x07" + b"onnx" + os.urandom(64) + b"\x00" * max(0, octets - 70)


# ---------------------------------------------------------------------------
def test_arbitrage_des_intentions():
    """L'anonymisation annule et remplace toute autre modification faciale.

    Ce test porte sur le mecanisme du greffon. Le worker en possede un second,
    redondant : le cas 6 de tests_worker.py le couvre separement, et
    tests_mutation.py verifie qu'aucun des deux ne masque l'absence de l'autre.
    """
    print("Arbitrage : table de preseance des intentions")
    bac = Bac()
    try:
        module = bac.module
        conflits = []
        retenues = module.arbitrer_intentions(
            ["anonymiser", "sourire", "ameliorer", "coloriser"], conflits)
        controler("anonymisation prioritaire : sourire et amelioration retires",
                  retenues == ["anonymiser", "coloriser"], str(retenues))
        controler("le conflit laisse une trace explicite",
                  len(conflits) == 2 and all("prioritaire" in c for c in conflits),
                  str(conflits))

        conflits = []
        retenues = module.arbitrer_intentions(["sourire", "ameliorer"], conflits)
        controler("sans anonymisation, les deux operations coexistent",
                  retenues == ["sourire", "ameliorer"] and not conflits,
                  str((retenues, conflits)))

        controler("la colorisation seule n'exige aucun detecteur",
                  module.roles_necessaires(["coloriser"]) == ["colorisation"],
                  str(module.roles_necessaires(["coloriser"])))
        controler("toute operation faciale exige le detecteur",
                  module.ROLE_DETECTION in module.roles_necessaires(["sourire"]),
                  str(module.roles_necessaires(["sourire"])))
    finally:
        bac.fermer()


def test_detection_lanceur_py():
    """Une strategie de detection qui ne trouve jamais rien ne se signale
    pas : elle doit donc etre couverte par un test sur une sortie reelle."""
    print("Detection : sortie reelle du lanceur py -0p")
    bac = Bac()
    try:
        import re
        sortie = open(os.path.join(SORTIES_REELLES, "py_0p_windows.txt"),
                      encoding="utf-8").read()
        trouves = re.findall(bac.module.MOTIF_LANCEUR_PY, sortie)
        controler("les quatre interpreteurs de la sortie reelle sont extraits",
                  len(trouves) == 4, str(trouves))
        controler("un chemin contenant des espaces n'est pas tronque",
                  "C:\\Program Files\\Python312\\python.exe" in trouves,
                  str(trouves))
        controler("le tag sans espace (-V:3.14) est reconnu",
                  any("Python314" in t for t in trouves), str(trouves))
    finally:
        bac.fermer()


def test_classement_des_interpreteurs():
    print("Interpreteur : classement par canal puis par version")
    bac = Bac()
    try:
        module = bac.module
        canal, score = module.classer_candidat(
            os.path.join("C:\\", "Program Files", "Blender", "python.exe"),
            module.CANAL_PATH)
        controler("un Python livre avec une application tierce est declasse",
                  score < module.CANAL_PATH[1] and "application_tierce" in canal,
                  "%s %s" % (canal, score))
        controler("mais jamais rejete : il garde un score positif", score >= 1,
                  str(score))

        candidats = [
            {"chemin": "a", "canal": "path", "score_canal": 30,
             "version": (3, 13, 0), "dans_plafond": 0},
            {"chemin": "b", "canal": "registre", "score_canal": 90,
             "version": (3, 9, 0), "dans_plafond": 1},
            {"chemin": "c", "canal": "registre", "score_canal": 90,
             "version": (3, 12, 0), "dans_plafond": 1},
        ]
        ordre = [d["chemin"] for d in module.trier_candidats(candidats)]
        controler("le canal declare au systeme passe avant la version",
                  ordre[0] == "c" and ordre[-1] == "a", str(ordre))
        controler("le plafond de version classe, il n'exclut pas",
                  "a" in ordre, str(ordre))
    finally:
        bac.fermer()


def test_emplacements_et_marqueur():
    print("Emplacements : donnees hors du profil, marqueur par greffon")
    bac = Bac()
    try:
        module = bac.module
        donnees = module.get_data_dir()
        controler("les donnees volumineuses sortent du profil GIMP",
                  not donnees.startswith(bac.profil), donnees)
        controler("aucun numero de version de GIMP dans le chemin de donnees",
                  "3.0" not in donnees and "3.2" not in donnees, donnees)
        controler("le venv est nomme par pile, sans nom de greffon",
                  module.nom_venv(module.STACK_CPU) == "venv-onnx-cpu",
                  module.nom_venv(module.STACK_CPU))
        controler("le marqueur porte le nom du greffon",
                  module.PLUGIN_ID in module.nom_marqueur(module.STACK_CPU),
                  module.nom_marqueur(module.STACK_CPU))
        controler("le fichier de confiance des modeles reste commun",
                  os.path.basename(module.get_tofu_path()) == "trusted_models.json",
                  module.get_tofu_path())

        module.ecrire_marqueur({"statut": "pret", "venv_python": sys.executable})
        marqueur = module.lire_marqueur()
        controler("le marqueur consigne le canal, l'horodatage et l'API",
                  marqueur.get("ecrit_le") and marqueur.get("api_gimp")
                  and marqueur.get("dossier_donnees"), str(marqueur))
        controler("un marqueur complet est juge utilisable",
                  module.marqueur_utilisable(marqueur), str(marqueur))
        module.invalider_marqueur("test")
        controler("un marqueur invalide ne l'est plus",
                  not module.marqueur_utilisable(module.lire_marqueur()))
    finally:
        bac.fermer()


def test_reprise_dossier_versionne():
    """Un dossier laisse par une version anterieure se reprend par renommage,
    jamais par un nouveau telechargement de plusieurs centaines de Mo."""
    print("Emplacements : reprise d'un dossier versionne par une version anterieure")
    bac = Bac()
    try:
        module = bac.module
        base = module.base_donnees()
        ancien = os.path.join(base, "GIMP", "3.0", module.SHARED_DIR_NAME)
        os.makedirs(os.path.join(ancien, "models"), exist_ok=True)
        temoin = os.path.join(ancien, "models", "temoin.onnx")
        with open(temoin, "wb") as f:
            f.write(faux_modele(2048))
        module._DOSSIER_DONNEES = None
        nouveau = module.get_data_dir()
        controler("le dossier versionne est repris a l'emplacement canonique",
                  os.path.isfile(os.path.join(nouveau, "models", "temoin.onnx")),
                  nouveau)
        controler("et il ne porte plus de numero de version",
                  nouveau.endswith(module.SHARED_DIR_NAME)
                  and "3.0" not in nouveau, nouveau)
    finally:
        bac.fermer()


def test_recherche_et_rangement_des_modeles():
    print("Modeles : recherche recursive, rangement, doublons inertes")
    bac = Bac()
    try:
        module = bac.module
        nom = module.MODELES[module.ROLE_DETECTION]["fichier"]
        sous = os.path.join(bac.modeles(), "yolo", "v8")
        os.makedirs(sous, exist_ok=True)
        contenu = faux_modele(2 * 1024 * 1024)
        with open(os.path.join(sous, nom), "wb") as f:
            f.write(contenu)
        chemin, origine = module.chercher_modele(nom)
        controler("un modele range dans un sous-dossier est trouve",
                  chemin is not None and origine == "sous_dossier_gere",
                  str((chemin, origine)))

        autre = os.path.join(bac.modeles(), "copie")
        os.makedirs(autre, exist_ok=True)
        with open(os.path.join(autre, nom), "wb") as f:
            f.write(contenu)
        range_ = module.ranger_modele(chemin, nom)
        controler("le modele remonte a l'emplacement canonique",
                  range_ == os.path.join(bac.modeles(), nom), range_)
        controler("le doublon de taille identique est supprime",
                  not os.path.isfile(os.path.join(autre, nom)))

        different = os.path.join(bac.modeles(), "autre_taille")
        os.makedirs(different, exist_ok=True)
        with open(os.path.join(different, nom), "wb") as f:
            f.write(faux_modele(3 * 1024 * 1024))
        module.ranger_modele(os.path.join(bac.modeles(), nom), nom)
        controler("un fichier de taille differente n'est pas touche",
                  os.path.isfile(os.path.join(different, nom)))
    finally:
        bac.fermer()


def test_vraisemblance_et_fichier_inutilisable():
    print("Modeles : controle de vraisemblance et telechargement interrompu")
    bac = Bac()
    try:
        module = bac.module
        nom = module.MODELES[module.ROLE_DETECTION]["fichier"]
        chemin = os.path.join(bac.modeles(), nom)

        for contenu, attendu in ((b"", "vide"), (b"<html>erreur</html>", "HTML"),
                                 (b"{\"erreur\": 1}", "JSON"),
                                 (b"version https://git-lfs.github.com/spec/v1",
                                  "LFS")):
            with open(chemin, "wb") as f:
                f.write(contenu)
            refuse = False
            try:
                module.controler_vraisemblance(chemin, nom)
            except ValueError:
                refuse = True
            controler("un fichier %s est refuse" % attendu, refuse)

        with open(chemin, "wb") as f:
            f.write(faux_modele(1024))
        avertissements = []
        raison = "tronque"
        supprime = module.ecarter_fichier_inutilisable(chemin, nom, raison,
                                                       avertissements)
        controler("un telechargement interrompu est ecarte du dossier gere",
                  supprime and not os.path.isfile(chemin))
        controler("et l'utilisateur est prevenu que le fichier a ete supprime",
                  avertissements and nom in avertissements[0],
                  str(avertissements))

        hors = os.path.join(bac.dossier, nom)
        with open(hors, "wb") as f:
            f.write(faux_modele(1024))
        controler("un fichier hors des dossiers geres n'est jamais supprime",
                  not module.ecarter_fichier_inutilisable(hors, nom, raison, [])
                  and os.path.isfile(hors))
    finally:
        bac.fermer()


def test_telechargement_annonce_et_seuil():
    print("Modeles : telechargement annonce, seuil, sources multiples")
    bac = Bac()
    serveur = None
    try:
        module = bac.module
        nom = module.MODELES[module.ROLE_DETECTION]["fichier"]
        petit = faux_modele(3 * 1024 * 1024)
        serveur = ServeurModeles({"ok.onnx": petit})
        module.MODELES[module.ROLE_DETECTION]["sources"] = [
            serveur.base + "absent.onnx",
            serveur.base + "ok.onnx",
        ]
        messages = []
        chemin, recus = module.telecharger_modele(module.ROLE_DETECTION,
                                                  messages.append)
        controler("une premiere adresse morte n'empeche pas la suivante",
                  os.path.isfile(chemin) and recus == len(petit),
                  str((chemin, recus)))
        controler("la taille est annoncee avant le transfert",
                  any("Telechargement" in m and "Mo" in m for m in messages),
                  str(messages[:3]))

        # Le serveur sert un petit fichier mais annonce une taille au-dela du
        # seuil : le refus doit venir du seuil, pas d'une adresse morte.
        serveur.fichiers["trop.onnx"] = faux_modele(1024)
        serveur.tailles_annoncees["trop.onnx"] = module.AUTO_DOWNLOAD_MAX_BYTES + 1
        module.MODELES[module.ROLE_DETECTION]["sources"] = [serveur.base + "trop.onnx"]
        avant = len(serveur.gets())
        refuse = False
        try:
            module.telecharger_modele(module.ROLE_DETECTION, messages.append)
        except RuntimeError:
            refuse = True
        controler("un fichier annonce au-dela du seuil est refuse", refuse)
        controler("aucun octet du corps n'est transfere lors d'un refus",
                  len(serveur.gets()) == avant,
                  str(serveur.requetes[-3:]))
    finally:
        if serveur:
            serveur.fermer()
        bac.fermer()


def test_degradation_des_modeles():
    print("Modeles : un modele absent degrade, il n'interrompt pas")
    bac = Bac()
    try:
        module = bac.module
        for role in module.MODELES:
            module.MODELES[role]["sources"] = []
        avertissements = []
        resolus = module.resoudre_modeles(
            [module.ROLE_DETECTION, module.ROLE_SOURIRE], True,
            lambda t: None, avertissements)
        controler("aucune exception n'est levee quand tout manque",
                  resolus == {module.ROLE_DETECTION: None,
                              module.ROLE_SOURIRE: None}, str(resolus))
        controler("le role avec repli annonce un chemin sans IA",
                  any("sans IA" in a for a in avertissements),
                  str(avertissements))
        controler("le role sans repli annonce une etape ignoree",
                  any("ignoree" in a for a in avertissements),
                  str(avertissements))

        message = module.message_depot_manuel([module.ROLE_SOURIRE])
        controler("le message de depot manuel donne un chemin collable",
                  module.get_models_dir() in message, message[:200])
        controler("et le nom exact du fichier attendu",
                  module.MODELES[module.ROLE_SOURIRE]["fichier"] in message)
    finally:
        bac.fermer()


def test_sources_utilisateur():
    print("Modeles : sources_modeles.json complete la table du code")
    bac = Bac()
    try:
        module = bac.module
        chemin = module.get_sources_utilisateur_path()
        with open(chemin, "w", encoding="utf-8") as f:
            json.dump({"sourire": "https://exemple.invalide/attgan.onnx"}, f)
        sources = module.sources_du_modele(module.ROLE_SOURIRE)
        controler("l'adresse de l'utilisateur passe devant celles du code",
                  sources and sources[0] == "https://exemple.invalide/attgan.onnx",
                  str(sources))
        with open(chemin, "w", encoding="utf-8") as f:
            f.write("ceci n'est pas du JSON")
        controler("un fichier illisible ne casse rien",
                  module.sources_du_modele(module.ROLE_SOURIRE)
                  == module.MODELES[module.ROLE_SOURIRE]["sources"])
    finally:
        bac.fermer()


def test_tofu():
    print("Modeles : confiance a la premiere utilisation")
    bac = Bac()
    try:
        module = bac.module
        avertissements = []
        module.tofu("modele.onnx", "aaa", avertissements)
        controler("la premiere empreinte est memorisee sans avertir",
                  not avertissements)
        module.tofu("modele.onnx", "aaa", avertissements)
        controler("une empreinte identique reste silencieuse", not avertissements)
        module.tofu("modele.onnx", "bbb", avertissements)
        controler("un changement d'empreinte avertit sans bloquer",
                  len(avertissements) == 1 and "change" in avertissements[0],
                  str(avertissements))
    finally:
        bac.fermer()


def test_refus_gpu_avant_telechargement():
    print("Materiel : l'option GPU est refusee avant tout transfert")
    bac = Bac()
    try:
        module = bac.module
        module._RUNTIME_CUDA = None
        present, detail = module.sonde_runtime_cuda()
        controler("la sonde porte sur le runtime, pas sur le pilote",
                  "cudart" in detail or "CUDA" in detail or "toolkit" in detail,
                  detail)
        message = module.refus_gpu(detail)
        controler("le refus explique ce qui manque", "runtime CUDA" in message)
        controler("le refus explique comment s'en passer",
                  "processeur" in message, message[:200])
        controler("le refus cite le constat plutot qu'une hypothese",
                  detail[:20] in message, message[:200])
    finally:
        bac.fermer()


def test_ecart_materiel_signale_une_fois():
    """Le message se declenche sur le symptome - le calcul s'est fait sur le
    processeur -, pas sur l'hypothese de sa cause, et seulement au regard de
    l'intention exprimee."""
    print("Materiel : ecart signale une seule fois, et seulement s'il en est un")
    bac = Bac()
    try:
        module = bac.module
        resultat = {"moteurs": [["detection", "yolo.onnx", "processeur"]],
                    "fournisseurs_disponibles": ["CPUExecutionProvider"]}

        avertissements = []
        module.signaler_ecart_materiel(False, resultat, avertissements)
        controler("sans demande de GPU, un calcul sur processeur ne dit rien",
                  not avertissements, str(avertissements))

        avertissements = []
        module.signaler_ecart_materiel(True, resultat, avertissements)
        controler("avec demande de GPU, l'ecart est signale",
                  len(avertissements) == 1, str(avertissements))
        controler("le message joint l'etat des deux couches",
                  "Runtime du systeme" in avertissements[0]
                  and "Fournisseurs du moteur" in avertissements[0],
                  avertissements[0][:200])

        avertissements = []
        module.signaler_ecart_materiel(True, resultat, avertissements)
        controler("il n'est pas repete au lancement suivant",
                  not avertissements, str(avertissements))

        avertissements = []
        gpu = {"moteurs": [["detection", "yolo.onnx", "GPU NVIDIA"]],
               "fournisseurs_disponibles": ["CUDAExecutionProvider"]}
        module.signaler_ecart_materiel(True, gpu, avertissements)
        controler("un calcul reellement accelere ne declenche rien",
                  not avertissements, str(avertissements))
        controler("le materiel constate est note dans le marqueur",
                  module.lire_marqueur().get("dernier_materiel_constate")
                  == "GPU NVIDIA", str(module.lire_marqueur()))
    finally:
        bac.fermer()


def test_verdict_acceleration_memorise():
    """Un echec d'acceleration deja constate ne se reconstate pas.

    C'est la section 17 du scenario : memoriser l'echec dans le marqueur pour
    ne pas relancer, a chaque ouverture du filtre, un travail dont on connait
    l'issue. Sans cela, les roues cuDNN etaient retentees et l'inference de
    controle rejouee a chaque lancement, pour aboutir au meme repli.
    """
    print("Materiel : un echec d'acceleration constate une fois ne se rejoue pas")
    bac = Bac()
    try:
        module = bac.module
        module.definir_pile(module.STACK_GPU)
        controler("aucun verdict au depart",
                  module.verdict_acceleration() is None)

        constat = {"ok": True, "fournisseur": "CPUExecutionProvider",
                   "detail": "", "disponibles": ["CPUExecutionProvider"],
                   "version": "1.17.0",
                   "journal": ["Failed to load library libonnxruntime_providers_cuda.so"]}
        module.memoriser_verdict_acceleration(False, constat)
        verdict = module.verdict_acceleration()
        controler("le verdict est memorise avec son constat",
                  verdict and verdict.get("verdict") == "echec"
                  and verdict.get("disponibles") == ["CPUExecutionProvider"],
                  str(verdict))

        # Un second lancement ne doit ni installer cuDNN ni rejouer l'inference.
        appels = []
        module.installer_cudnn = lambda *a, **k: appels.append("cudnn") or (False, "")
        module.valider_acceleration = lambda *a, **k: appels.append("validation") or {}
        module.preparer_environnement = lambda *a, **k: appels.append("env") or sys.executable
        greffon = module.IaVisageStudioPlugin()
        avertissements = []
        module.definir_pile(module.STACK_GPU)
        # Un modele doit etre fourni, sinon le code s'arrete avant l'inference
        # de controle pour une tout autre raison, et le controle ci-dessous ne
        # prouverait rien.
        modeles = {module.ROLE_DETECTION_YUNET:
                   os.path.join(bac.dossier, "factice.onnx")}
        utiliser, python, env, fournisseurs = greffon._valider_ou_degrader(
            sys.executable, modeles, bac.dossier, lambda t: None,
            avertissements, False)
        controler("aucune roue cuDNN n'est retentee", "cudnn" not in appels,
                  str(appels))
        controler("l'inference de controle n'est pas rejouee",
                  "validation" not in appels, str(appels))
        controler("le traitement bascule sur le processeur",
                  utiliser is False and fournisseurs == module.FOURNISSEURS_CPU,
                  str((utiliser, fournisseurs)))
        message = "\n".join(avertissements)
        controler("le message dit que le constat est memorise",
                  "memorise" in message or "constat deja etabli" in message,
                  message[:200])
        controler("le message joint l'etat des deux couches",
                  "Fournisseurs declares par le moteur" in message
                  and "Runtime du systeme" in message, message[:300])
        controler("le message cite ce que le moteur a dit lui-meme",
                  "libonnxruntime_providers_cuda" in message, message[:400])
        controler("le message dit comment refaire l'essai",
                  "Reinstaller l'environnement IA" in message, message[:600])
    finally:
        bac.fermer()


def test_cause_probable():
    """La cause s'etablit sur la trace du moteur, elle ne s'affirme pas.

    La premiere version de ce message accusait cuDNN de confiance. Le journal
    de l'utilisateur disait tout autre chose : une liste de fournisseurs
    rejetee en bloc parce qu'elle contenait un nom inconnu de sa build - un
    defaut du greffon, pas de son poste.
    """
    print("Materiel : la cause annoncee s'appuie sur la trace du moteur")
    bac = Bac()
    try:
        module = bac.module
        rejet = {"disponibles": ["CUDAExecutionProvider", "CPUExecutionProvider"],
                 "journal": ["EP Error Unknown Provider Type: "
                             "ROCMExecutionProvider when using [...]"]}
        phrase = module.cause_probable(rejet)
        controler("une liste rejetee en bloc est nommee comme telle",
                  "liste de fournisseurs" in phrase, phrase)
        controler("et elle n'accuse pas cuDNN", "cuDNN" not in phrase, phrase)

        cudnn = {"disponibles": ["CUDAExecutionProvider", "CPUExecutionProvider"],
                 "journal": ["Failed to load library "
                             "onnxruntime_providers_cuda.dll: cudnn64_8.dll"]}
        controler("un vrai probleme de cuDNN, lui, est nomme",
                  "cuDNN" in module.cause_probable(cudnn),
                  module.cause_probable(cudnn))

        muet = {"disponibles": ["CUDAExecutionProvider", "CPUExecutionProvider"],
                "journal": []}
        controler("sans indice, le message ne suppose rien",
                  "n'en dit pas la raison" in module.cause_probable(muet),
                  module.cause_probable(muet))
    finally:
        bac.fermer()


def test_verdict_perime_par_une_nouvelle_version():
    """Un verdict rendu par une version anterieure ne vaut plus.

    La validation elle-meme peut avoir ete corrigee - c'est arrive. Sans cette
    peremption, un poste dont l'acceleration marche resterait sur le
    processeur indefiniment, sur la foi d'un constat errone.
    """
    print("Materiel : un verdict d'acceleration se perime avec le greffon")
    bac = Bac()
    try:
        module = bac.module
        module.definir_pile(module.STACK_GPU)
        module.memoriser_verdict_acceleration(
            False, {"fournisseur": "CPUExecutionProvider", "disponibles": []})
        controler("le verdict vaut pour la version courante",
                  module.verdict_acceleration() is not None)
        marqueur = module.lire_marqueur()
        marqueur["acceleration"]["version_greffon"] = "0.9"
        module.ecrire_marqueur(marqueur)
        controler("un verdict d'une version anterieure est ignore",
                  module.verdict_acceleration() is None,
                  str(module.lire_marqueur().get("acceleration")))
    finally:
        bac.fermer()


def test_diagnostic_du_moteur():
    print("Materiel : extraction du diagnostic ecrit par le moteur")
    bac = Bac()
    try:
        module = bac.module
        brut = ("2026-09-17 10:00:00 [I] Creating session\n"
                "[E:onnxruntime] Failed to load library "
                "libonnxruntime_providers_cuda.so: libcudnn.so.8: cannot open "
                "shared object file\n"
                "une ligne sans rapport\n"
                "[W] Falling back to CPUExecutionProvider\n")
        lignes = module.lignes_diagnostic_moteur(brut)
        controler("les lignes qui nomment la cause sont retenues",
                  any("libcudnn" in l for l in lignes), str(lignes))
        controler("les lignes sans rapport sont ecartees",
                  not any("sans rapport" in l for l in lignes), str(lignes))
        controler("la queue est bornee", len(lignes) <= 4, str(len(lignes)))
    finally:
        bac.fermer()


def test_annulation_et_processus_orphelins():
    print("Processus : annulation, et aucun enfant ne survit au parent")
    bac = Bac()
    try:
        module = bac.module
        module._SONDE_ANNULATION_FAITE = False
        controler("sans sonde exposee par la build, aucune annulation n'est "
                  "supposee", module.sonde_annulation() is None
                  and not module.annulation_demandee())

        faux_gimp.installer_sonde_annulation(True)
        module._SONDE_ANNULATION_FAITE = False
        controler("quand la build expose une sonde, elle est utilisee",
                  module.sonde_annulation() is not None
                  and module.annulation_demandee())

        proc = module.demarrer_processus(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            module.clean_env())
        code, depasse, annule = module.attendre_processus(proc, 30, "attente")
        controler("un processus est tue des que l'annulation est demandee",
                  annule and not depasse, str((code, depasse, annule)))
        controler("et il est reellement mort", proc.poll() is not None)

        faux_gimp.retirer_sonde_annulation()
        module._SONDE_ANNULATION_FAITE = False
        orphelin = module.demarrer_processus(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            module.clean_env())
        tues = module.tuer_processus_restants()
        temps = time.time()
        while orphelin.poll() is None and time.time() - temps < 10:
            time.sleep(0.1)
        controler("le filet de securite tue ce qui reste a la sortie",
                  tues == 1 and orphelin.poll() is not None,
                  str((tues, orphelin.poll())))
    finally:
        bac.fermer()


def test_isolation_environnement():
    print("Environnement : purge des variables et filtrage du PATH")
    bac = Bac()
    try:
        module = bac.module
        sauve = {}
        for nom in ("PYTHONPATH", "LD_PRELOAD", "GI_TYPELIB_PATH", "PATH"):
            sauve[nom] = os.environ.get(nom)
        os.environ["PYTHONPATH"] = "/gimp/python"
        os.environ["LD_PRELOAD"] = "/gimp/lib/libfoo.so"
        os.environ["GI_TYPELIB_PATH"] = "/gimp/typelib"
        os.environ["PATH"] = os.pathsep.join(
            ["/usr/bin", "/opt/GIMP/bin", "/usr/local/cuda/bin"])
        env = module.clean_env()
        controler("les variables toxiques sont purgees",
                  not any(n in env for n in ("PYTHONPATH", "LD_PRELOAD",
                                             "GI_TYPELIB_PATH")),
                  str([n for n in ("PYTHONPATH", "LD_PRELOAD", "GI_TYPELIB_PATH")
                       if n in env]))
        controler("PYTHONNOUSERSITE et PYTHONUTF8 sont poses",
                  env.get("PYTHONNOUSERSITE") == "1"
                  and env.get("PYTHONUTF8") == "1", str(env.get("PYTHONUTF8")))
        chemins = env["PATH"].split(os.pathsep)
        controler("les entrees de PATH contenant gimp sont retirees",
                  not any("gimp" in c.lower() for c in chemins), str(chemins))
        controler("le filtrage reste chirurgical : le runtime CUDA reste sur "
                  "le PATH", "/usr/local/cuda/bin" in chemins, str(chemins))

        dossier_dll = os.path.join(bac.dossier, "nvidia", "bin")
        os.makedirs(dossier_dll, exist_ok=True)
        env = module.clean_env([dossier_dll])
        controler("les dossiers de DLL demandes sont exposes en tete de PATH",
                  env["PATH"].split(os.pathsep)[0] == dossier_dll,
                  env["PATH"][:120])

        for nom, valeur in sauve.items():
            if valeur is None:
                os.environ.pop(nom, None)
            else:
                os.environ[nom] = valeur
    finally:
        bac.fermer()


def test_espace_disque():
    print("Disque : le controle n'a lieu qu'avant une installation engagee")
    bac = Bac()
    try:
        module = bac.module
        refuse = False
        try:
            module.verifier_espace(bac.dossier, 1 << 60, "un test")
        except RuntimeError as e:
            refuse = True
            message = str(e)
        controler("un besoin manifestement hors de portee est refuse", refuse)
        if refuse:
            controler("le message chiffre le requis, le disponible et le chemin",
                      "Requis" in message and "Disponible" in message
                      and bac.dossier in message, message[:200])
        ok = True
        try:
            module.verifier_espace(bac.dossier, 1024, "un test")
        except RuntimeError:
            ok = False
        controler("un besoin modeste passe", ok)
    finally:
        bac.fermer()


def test_journaux_et_messages():
    print("Journaux : archivage hors du dossier de travail, messages bornes")
    bac = Bac()
    try:
        module = bac.module
        travail = tempfile.mkdtemp(prefix="travail_")
        for nom in ("worker.log", "cfg.json", "worker.py", "image.png"):
            with open(os.path.join(travail, nom), "w") as f:
                f.write("contenu")
        archive = module.archiver_journaux(travail, {"operations": ["anonymiser"]})
        controler("les journaux sont archives hors du dossier de travail",
                  archive and os.path.isdir(archive), str(archive))
        contenu = sorted(os.listdir(archive))
        controler("journaux, parametres et worker sont conserves",
                  {"worker.log", "cfg.json", "worker.py"} <= set(contenu),
                  str(contenu))
        controler("l'image, elle, ne l'est pas", "image.png" not in contenu,
                  str(contenu))
        incident = json.load(open(os.path.join(archive, "incident.json"),
                                  encoding="utf-8"))
        controler("l'incident nomme le greffon qui l'a produit",
                  incident.get("greffon") == module.PLUGIN_ID, str(incident))
        shutil.rmtree(travail, ignore_errors=True)

        # Un greffon voisin de la suite depose ses archives dans le meme
        # dossier, avec une autre convention de nommage. En ASCII le tiret
        # precede le chiffre : un tri alphabetique placerait toujours cette
        # forme en tete, et une purge qui s'y fierait la supprimerait en
        # premier, quel que soit son age.
        voisines = []
        for heure in ("19-44-05", "19-44-06", "19-44-07"):
            d = os.path.join(module.get_logs_dir(), "2026-09-16_" + heure)
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, "incident.json"), "w",
                      encoding="utf-8") as f:
                json.dump({"greffon": "ia_detourage", "version": "3.11.0"}, f)
            voisines.append(d)
        sans_marque = os.path.join(module.get_logs_dir(), "2026-01-01_00-00-00")
        os.makedirs(sans_marque, exist_ok=True)

        for _ in range(module.ARCHIVES_A_CONSERVER + 4):
            t = tempfile.mkdtemp(prefix="travail_")
            with open(os.path.join(t, "worker.log"), "w") as f:
                f.write("x")
            module.archiver_journaux(t)
            shutil.rmtree(t, ignore_errors=True)
        miennes = module.archives_du_greffon(module.get_logs_dir())
        controler("la purge conserve les dix incidents de ce greffon",
                  len(miennes) == module.ARCHIVES_A_CONSERVER, str(len(miennes)))
        controler("les archives d'un greffon voisin sont intactes",
                  all(os.path.isdir(d) for d in voisines),
                  str([d for d in voisines if not os.path.isdir(d)]))
        controler("une archive recente non identifiee est conservee",
                  os.path.isdir(sans_marque), sans_marque)

        # Le stock orphelin - archives d'avant le marqueur, ou d'un greffon de
        # la suite qui ne le pose pas encore - se resorbe, mais seulement au
        # dela de deux conditions reunies : l'age et le nombre.
        jour = 86400
        anciennes = []
        for index in range(module.ARCHIVES_A_CONSERVER + 3):
            d = os.path.join(module.get_logs_dir(), "2025-01-%02d_00-00-00"
                             % (index + 1))
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, "worker.log"), "w") as f:
                f.write("x")
            quand = time.time() - 40 * jour
            os.utime(os.path.join(d, "worker.log"), (quand, quand))
            os.utime(d, (quand, quand))
            anciennes.append(d)
        module.purger_journaux(module.get_logs_dir())
        # sans_marque compte lui aussi parmi les orphelines, et il est le plus
        # recent : le decompte porte donc sur l'ensemble, pas sur les seules
        # anciennes.
        orphelines = [d for d in os.listdir(module.get_logs_dir())
                      if os.path.isdir(os.path.join(module.get_logs_dir(), d))
                      and not os.path.isfile(os.path.join(
                          module.get_logs_dir(), d, "incident.json"))]
        supprimees = [d for d in anciennes if not os.path.isdir(d)]
        controler("les archives orphelines anciennes se resorbent",
                  len(orphelines) == module.ARCHIVES_A_CONSERVER
                  and len(supprimees) == 4,
                  "%d orphelines restantes, %d anciennes supprimees"
                  % (len(orphelines), len(supprimees)))
        controler("celle d'un voisin, recente, n'a toujours pas bouge",
                  all(os.path.isdir(d) for d in voisines)
                  and os.path.isdir(sans_marque), str(voisines))

        texte = ("RAISON: le modele est absent.\n"
                 "Detail : 400 lignes de sortie pip\nencore une ligne")
        controler("un message repris ailleurs ne traine pas sa queue de journal",
                  module.premiere_phrase(texte) == "le modele est absent.",
                  module.premiere_phrase(texte))
        controler("sans prefixe repere, seule la premiere phrase est reprise",
                  module.premiere_phrase("Echec du truc. Et plein de details")
                  == "Echec du truc",
                  module.premiere_phrase("Echec du truc. Et plein de details"))

        chemin = os.path.join(bac.dossier, "journal_utf16.log")
        with open(chemin, "wb") as f:
            f.write("erreur accentuee".encode("utf-16"))
        controler("un journal en UTF-16 reste lisible",
                  "erreur" in module.lire_journal(chemin),
                  repr(module.lire_journal(chemin)[:40]))
    finally:
        bac.fermer()


def _venv_factice(module, paquets):
    """Cree un venv minimal et y depose des modules bouchons.

    Sans pip : le but est justement de verifier qu'il n'est jamais appele. Les
    bouchons rendent le test independant de ce qui est installe sur la machine,
    et c'est le chemin de decision qui est teste, pas les vraies roues.
    """
    venv = os.path.join(module.get_data_dir(), module.nom_venv(module.STACK_CPU))
    code = subprocess.call([sys.executable, "-m", "venv", "--without-pip", venv],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if code != 0:
        return None, None, None
    py = module.chemin_python_venv(venv)
    site = subprocess.run(
        [py, "-c", "import site; print(site.getsitepackages()[0])"],
        capture_output=True, text=True).stdout.strip()
    if not site or not os.path.isdir(site):
        return None, None, None
    for nom in paquets:
        with open(os.path.join(site, nom + ".py"), "w", encoding="utf-8") as f:
            f.write("__version__ = '0.0-bouchon'\n")
    return venv, py, site


def test_poste_deja_installe():
    """L'etat que les tests oublient le plus souvent : celui d'un poste deja
    installe.

    Le venv de la pile est la, pose par un autre greffon de la suite, et ce
    greffon-ci n'a pas encore de marqueur. Il doit constater et demarrer, pas
    reinstaller plusieurs centaines de megaoctets. Le second cas verifie que ce
    test sait dire le contraire : un venv incomplet doit, lui, declencher une
    reparation.
    """
    print("Environnement : poste ou un autre greffon a deja installe la pile")
    bac = Bac()
    try:
        module = bac.module
        venv, py, site = _venv_factice(module, ("numpy", "cv2", "onnxruntime"))
        if not py:
            controler("un venv de test a pu etre cree", False,
                      "creation impossible sur ce poste")
            return
        controler("aucun marqueur pour ce greffon au depart",
                  not os.path.isfile(module.get_marker_path()))

        lances = []
        original = module.demarrer_processus

        def espion(cmd, env, fichier_log=None):
            lances.append(list(cmd))
            return original(cmd, env, fichier_log)

        module.demarrer_processus = espion
        # La decouverte d'interpreteur a ses propres tests ; on la neutralise
        # ici pour que ce test ne porte que sur la decision de reinstaller.
        module.decouvrir_pythons = lambda: [
            {"chemin": sys.executable, "canal": "test", "version": (3, 11, 0),
             "score_canal": 90, "dans_plafond": 1}]

        travail = tempfile.mkdtemp(prefix="travail_")
        # L'appel est protege : un greffon qui reinstallerait ici echouerait
        # faute de pip, et une exception qui remonte ferait sauter les
        # controles suivants - le test dirait alors moins que ce qu'il sait.
        retour, incident = None, ""
        try:
            retour = module.preparer_environnement(module.STACK_CPU, travail,
                                                   False, lambda texte: None)
        except Exception as e:
            incident = str(e)[:200]
        pip = [c for c in lances if "pip" in " ".join(c)]
        creations = [c for c in lances if "venv" in c]
        controler("l'interpreteur du venv existant est retenu",
                  retour is not None
                  and os.path.normpath(retour) == os.path.normpath(py),
                  retour or incident)
        controler("aucune commande pip n'est lancee", not pip, str(pip))
        controler("aucun venv n'est recree", not creations, str(creations))
        marqueur = module.lire_marqueur()
        controler("le greffon ecrit son propre marqueur, sans toucher a celui "
                  "du voisin",
                  marqueur.get("statut") == "pret"
                  and module.PLUGIN_ID in os.path.basename(module.get_marker_path()),
                  str(marqueur.get("statut")))

        # Second cas : le venv existe mais lui manque une dependance. Le
        # greffon doit alors reparer, faute de quoi le controle ci-dessus ne
        # prouverait rien - il serait vert meme si plus rien n'etait jamais
        # installe.
        os.remove(os.path.join(site, "onnxruntime.py"))
        module.invalider_marqueur("test : dependance retiree")
        lances[:] = []
        try:
            module.preparer_environnement(module.STACK_CPU, travail, False,
                                          lambda texte: None)
        except Exception:
            pass
        pip = [c for c in lances if "pip" in " ".join(c)]
        controler("un venv incomplet declenche bien une reparation", bool(pip),
                  str(lances))
        shutil.rmtree(travail, ignore_errors=True)
    finally:
        bac.fermer()


def test_api_gimp():
    print("API GIMP : lecture d'options et de tuples sans dependre des indices")
    bac = Bac()
    try:
        module = bac.module

        class Config:
            def get_property(self, nom):
                if nom == "existe":
                    return 7
                raise AttributeError(nom)

        controler("une propriete absente retourne le defaut",
                  module.lire_option(Config(), "absente", 3) == 3)
        controler("une propriete presente est lue",
                  module.lire_option(Config(), "existe", 3) == 7)

        class Image:
            @staticmethod
            def get_width():
                return 800

            @staticmethod
            def get_height():
                return 600

        for retour, attendu in (((True, 10, 20, 110, 220), (True, 10, 20, 100, 200)),
                                ((10, 20, 110, 220), (True, 10, 20, 100, 200)),
                                ((False, 0, 0, 0, 0), (False, 0, 0, 800, 600))):
            faux_gimp.FauxGimp.Selection.retour = retour
            controler("bornes de selection lues sans dependre des indices : %s"
                      % (retour,),
                      module.bornes_selection(Image()) == attendu,
                      str(module.bornes_selection(Image())))
        faux_gimp.FauxGimp.Selection.retour = None
        controler("une API indisponible retombe sur l'image entiere",
                  module.bornes_selection(Image()) == (False, 0, 0, 800, 600),
                  str(module.bornes_selection(Image())))
    finally:
        bac.fermer()


def test_etiquette_du_calque():
    """Le journal du worker disparait avec le dossier temporaire : le nom du
    calque est la seule trace persistante."""
    print("Resultat : le nom du calque nomme le moteur et le materiel")
    bac = Bac()
    try:
        module = bac.module
        resultat = {"moteurs": [["detection", "yolov8n-face.onnx", "processeur"],
                                ["anonymisation", "OpenCV", "processeur"]]}
        etiquette = module.etiquette_resultat(resultat)
        controler("le moteur reellement employe figure dans le nom",
                  "yolov8n-face.onnx" in etiquette and "OpenCV" in etiquette,
                  etiquette)
        controler("le materiel de calcul aussi, en termes comprehensibles",
                  "processeur" in etiquette, etiquette)
        controler("sans traitement constate, le nom le dit",
                  "aucun" in module.etiquette_resultat({}),
                  module.etiquette_resultat({}))
    finally:
        bac.fermer()


def main():
    print("Tests du greffon IA Visage Studio (doublure de l'API GIMP)")
    print("")
    test_arbitrage_des_intentions()
    test_detection_lanceur_py()
    test_classement_des_interpreteurs()
    test_emplacements_et_marqueur()
    test_reprise_dossier_versionne()
    test_recherche_et_rangement_des_modeles()
    test_vraisemblance_et_fichier_inutilisable()
    test_telechargement_annonce_et_seuil()
    test_degradation_des_modeles()
    test_sources_utilisateur()
    test_tofu()
    test_refus_gpu_avant_telechargement()
    test_ecart_materiel_signale_une_fois()
    test_verdict_acceleration_memorise()
    test_diagnostic_du_moteur()
    test_cause_probable()
    test_verdict_perime_par_une_nouvelle_version()
    test_annulation_et_processus_orphelins()
    test_isolation_environnement()
    test_espace_disque()
    test_poste_deja_installe()
    test_journaux_et_messages()
    test_api_gimp()
    test_etiquette_du_calque()
    print("")
    echecs = [nom for nom, ok, _ in RESULTATS if not ok]
    print("%d controles, %d echec(s)" % (len(RESULTATS), len(echecs)))
    for nom in echecs:
        print("  ECHEC : " + nom)
    return 1 if echecs else 0


if __name__ == "__main__":
    sys.exit(main())
