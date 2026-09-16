#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Tests de bout en bout du worker embarque dans le greffon.

Usage : python3 outils/tests_worker.py   (necessite numpy et opencv)

Aucun modele reel n'est telecharge : onnxruntime est remplace par un double
qui repond en fonction de son entree, et non toujours la meme chose. Un double
constant ne testerait que la plomberie - que les fichiers circulent et que le
code ne leve pas - et ne dirait rien de la semantique. Ici, il fait deux
choses qu'aucun autre dispositif ne fait :

  - **il verifie le contrat** de chaque modele. Le faux detecteur exige une
    entree 1x3x640x640 en RGB divisee par 255 ; les faux modeles de visage
    exigent une entree normalisee entre -1 et 1, ce qu'une simple division par
    255 ne produit pas. Le vrai modele, lui, accepterait sans broncher et
    rendrait des resultats mediocres.
  - **il est geometrique**. Le faux detecteur calcule lui-meme la mise en
    lettre-boite et rend des boites dans cet espace ; le worker doit refaire
    le chemin inverse. Un decalage de quelques dizaines de pixels ne produit
    aucune erreur, seulement un flou a cote du visage - c'est exactement le
    defaut que ce double rend visible.

La validation porte sur des artefacts produits - image de sortie, fichier
temoin, code de retour, position et aire des zones modifiees - jamais sur la
presence d'un texte dans la sortie du processus.
"""

import ast
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import cv2

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FICHIER_GREFFON = os.path.join(RACINE, "ia_visage_studio.py")

# Fichiers factices : le worker ne lit jamais leur contenu, c'est le double
# d'onnxruntime qui decide du comportement d'apres leur nom.
MODELES_FACTICES = {
    "detection": "yolov8n-face.onnx",
    "sourire": "attgan_smile.onnx",
    "amelioration": "gfpgan_1_4.onnx",
    "colorisation": "deoldify_artistic.onnx",
}

FAUX_ONNXRUNTIME = r'''
import json
import os

import numpy as np

TRACE = os.environ["TEST_TRACE"]
VISAGES = json.loads(os.environ.get("TEST_VISAGES", "[]"))
TAILLE_IMAGE = json.loads(os.environ.get("TEST_TAILLE_IMAGE", "[640, 640]"))
ECHEC = os.environ.get("TEST_ECHEC", "")
ABERRANT = os.environ.get("TEST_SORTIE_ABERRANTE", "")
TEINTE = json.loads(os.environ.get("TEST_TEINTE", "[0.9, 0.5, 0.2]"))
SOURCE_GRISE = os.environ.get("TEST_SOURCE_GRISE") == "1"
DISPOSITION = os.environ.get("TEST_DISPOSITION", "canaux_dabord")
COLONNES_TOTALES = int(os.environ.get("TEST_COLONNES_TOTALES", "8400"))


def tracer(evenement):
    with open(TRACE, "a") as f:
        f.write(json.dumps(evenement) + chr(10))


def role_du_chemin(chemin):
    nom = os.path.basename(chemin).lower()
    if "yolo" in nom:
        return "detection"
    if "attgan" in nom:
        return "sourire"
    if "gfpgan" in nom:
        return "amelioration"
    if "deoldify" in nom:
        return "colorisation"
    return "inconnu"


def get_available_providers():
    return json.loads(os.environ.get("TEST_FOURNISSEURS",
                                     '["CPUExecutionProvider"]'))


class SessionOptions(object):
    def __init__(self):
        self.enable_mem_pattern = True
        self.enable_cpu_mem_arena = True


class _Info(object):
    def __init__(self, name, shape=None):
        self.name = name
        self.shape = shape or []


def _lettre_boite(x, y, largeur, hauteur, taille):
    """Meme calcul que le worker, mais dans le sens direct.

    Le worker doit retrouver (x, y, largeur, hauteur) a partir de ce que cette
    fonction produit. S'il oublie le decalage ou l'echelle, la zone modifiee
    atterrit ailleurs, sans la moindre erreur.
    """
    limage, himage = TAILLE_IMAGE
    echelle = min(float(taille) / float(limage), float(taille) / float(himage))
    nl = max(1, int(round(limage * echelle)))
    nh = max(1, int(round(himage * echelle)))
    dx = (taille - nl) // 2
    dy = (taille - nh) // 2
    cx = (x + largeur / 2.0) * echelle + dx
    cy = (y + hauteur / 2.0) * echelle + dy
    return cx, cy, largeur * echelle, hauteur * echelle


def _verifier_zero_un(tenseur, role):
    mini, maxi = float(tenseur.min()), float(tenseur.max())
    assert mini >= -0.01, "%s: entree normalisee entre -1 et 1 au lieu de 0 et 1 (min=%r)" % (role, mini)
    assert maxi <= 1.01, "%s: entree hors de [0, 1] (max=%r)" % (role, maxi)


def _verifier_moins_un_un(tenseur, role):
    mini, maxi = float(tenseur.min()), float(tenseur.max())
    assert mini < -0.1, "%s: entree non normalisee entre -1 et 1 (min=%r)" % (role, mini)
    assert maxi <= 1.01, "%s: entree hors de [-1, 1] (max=%r)" % (role, maxi)


def _verifier_rgb(tenseur, role):
    """Le pivot est en RGB. L'image de test est nettement rouge : si les
    canaux ont ete permutes en chemin, le bleu passe devant.

    Le controle n'a de sens que sur une source couleur : une image grise a,
    par construction, autant de rouge que de bleu.
    """
    if SOURCE_GRISE:
        return
    rouge = float(tenseur[0, 0].mean())
    bleu = float(tenseur[0, 2].mean())
    assert rouge > bleu + 0.02, \
        "%s: canaux permutes, rouge=%r bleu=%r (BGR au lieu de RGB ?)" % (role, rouge, bleu)


class InferenceSession(object):
    def __init__(self, path, sess_options=None, providers=None, **kwargs):
        self.role = role_du_chemin(path)
        if ECHEC == "modele:" + self.role:
            raise RuntimeError("chargement refuse par le scenario de test")
        self.providers = list(providers or ["CPUExecutionProvider"])
        tracer({"evenement": "session", "role": self.role,
                "providers": self.providers,
                "mem_pattern": getattr(sess_options, "enable_mem_pattern", None),
                "cpu_arena": getattr(sess_options, "enable_cpu_mem_arena", None)})

    def __del__(self):
        # Trace de la liberation reelle. C'est le seul moyen de verifier que
        # le worker lache sa reference AVANT la pause, et non apres : une
        # fonction de nettoyage qui ferait "del" sur son propre parametre
        # laisserait la session vivante pendant toute la purge.
        try:
            tracer({"evenement": "fermeture", "role": self.role})
        except Exception:
            pass

    def get_providers(self):
        return list(self.providers)

    def get_inputs(self):
        if self.role == "detection":
            return [_Info("images", [1, 3, 640, 640])]
        if self.role == "sourire":
            return [_Info("image", [1, 3, 256, 256]), _Info("attributs", [1, 13])]
        if self.role == "amelioration":
            return [_Info("input", [1, 3, 512, 512])]
        return [_Info("input", [1, 3, 256, 256])]

    def get_outputs(self):
        return [_Info("output")]

    def run(self, noms, alimentation):
        if ECHEC == "inference:" + self.role:
            raise RuntimeError("calcul refuse par le scenario de test")
        tenseur = np.asarray(list(alimentation.values())[0], dtype=np.float32)
        assert tenseur.ndim == 4 and tenseur.shape[1] == 3, \
            "%s: forme %r" % (self.role, tenseur.shape)
        if ABERRANT == self.role:
            return [np.zeros((1, 10), dtype=np.float32)]

        if self.role == "detection":
            assert tenseur.shape[2:] == (640, 640), \
                "detection: taille %r au lieu de 640x640" % (tenseur.shape[2:],)
            _verifier_zero_un(tenseur, "detection")
            _verifier_rgb(tenseur, "detection")
            tracer({"evenement": "inference", "role": "detection",
                    "min": float(tenseur.min()), "max": float(tenseur.max())})
            colonnes = []
            for x, y, largeur, hauteur, score in VISAGES:
                cx, cy, bl, bh = _lettre_boite(x, y, largeur, hauteur, 640)
                colonnes.append([cx, cy, bl, bh, score])
            # Un vrai export rend plusieurs milliers de propositions, dont la
            # quasi-totalite sous le seuil de score. Les omettre donnerait un
            # banc plus commode que la realite.
            while len(colonnes) < COLONNES_TOTALES:
                colonnes.append([0.0, 0.0, 1.0, 1.0, 0.0])
            tableau = np.asarray(colonnes, dtype=np.float32)
            if DISPOSITION == "canaux_dabord":
                tableau = tableau.T
            return [tableau[np.newaxis, ...]]

        _verifier_moins_un_un(tenseur, self.role)
        if self.role != "colorisation":
            _verifier_rgb(tenseur, self.role)
        tracer({"evenement": "inference", "role": self.role,
                "taille": list(tenseur.shape[2:]),
                "entrees": sorted(alimentation.keys())})

        if self.role == "colorisation":
            # Le colorisateur rend une teinte franche et uniforme : le test
            # peut alors verifier que la luminance vient bien de l'original et
            # la chrominance du modele.
            sortie = np.zeros_like(tenseur)
            for canal in range(3):
                sortie[:, canal, :, :] = TEINTE[canal] * 2.0 - 1.0
            return [sortie]

        # Sourire et amelioration : on marque la vignette d'une facon
        # reconnaissable, mais qui depend de l'entree - la moitie gauche est
        # eclaircie, la droite assombrie. Un double constant ne dirait rien de
        # l'orientation ni de la mise a l'echelle.
        sortie = tenseur.copy()
        moitie = sortie.shape[3] // 2
        sortie[:, :, :, :moitie] = np.clip(sortie[:, :, :, :moitie] + 0.5, -1.0, 1.0)
        sortie[:, :, :, moitie:] = np.clip(sortie[:, :, :, moitie:] - 0.5, -1.0, 1.0)
        return [sortie]
'''


def extraire_constante(nom):
    """Constantes relues dans le greffon, jamais recopiees ici.

    Un seuil recopie derive, et un test qui s'execute avec d'autres valeurs que
    celles livrees ne prouve rien.
    """
    source = open(FICHIER_GREFFON, encoding="utf-8").read()
    for noeud in ast.parse(source).body:
        if not isinstance(noeud, ast.Assign) or len(noeud.targets) != 1:
            continue
        cible = noeud.targets[0]
        if getattr(cible, "id", "") != nom:
            continue
        return ast.literal_eval(noeud.value)
    raise SystemExit("constante %s introuvable dans le greffon" % nom)


def extraire_worker():
    source = open(FICHIER_GREFFON, encoding="utf-8").read()
    for noeud in ast.parse(source).body:
        if isinstance(noeud, ast.Assign):
            for cible in noeud.targets:
                if getattr(cible, "id", "") == "WORKER_SCRIPT":
                    return ast.literal_eval(noeud.value)
    raise SystemExit("WORKER_SCRIPT introuvable")


def cles_lues_par_le_worker():
    worker = extraire_worker()
    cles = set(re.findall(r'cfg\.get\(\s*"([a-z_]+)"', worker))
    cles |= set(re.findall(r'cfg\[\s*"([a-z_]+)"\s*\]', worker))
    return cles


def verifier_couverture_des_cles(parametres):
    """Un banc de test qui oublie une cle fait retomber le worker sur sa
    valeur par defaut. Le test s'execute alors avec d'autres reglages que ceux
    livres et ne prouve plus rien - c'est arrive, d'ou ce refus de demarrer."""
    manquantes = cles_lues_par_le_worker() - set(parametres)
    if manquantes:
        raise SystemExit("cles absentes du dictionnaire de test : %s"
                         % ", ".join(sorted(manquantes)))


def png_gris_alpha(chemin, tableau):
    """Ecrit un PNG gris + alpha, qu'OpenCV ne sait pas produire.

    GIMP, lui, en exporte des que l'image est en niveaux de gris avec
    transparence : c'est un des quatre cas de canaux a couvrir, et le sauter
    reviendrait a ne tester que ce qui est commode.
    """
    import struct
    import zlib

    hauteur, largeur = tableau.shape[:2]
    seize = tableau.dtype == np.uint16
    profondeur = 16 if seize else 8
    lignes = bytearray()
    for y in range(hauteur):
        lignes.append(0)
        if seize:
            lignes.extend(tableau[y].astype(">u2").tobytes())
        else:
            lignes.extend(tableau[y].astype(np.uint8).tobytes())

    def morceau(nom, donnees):
        bloc = nom + donnees
        return (struct.pack(">I", len(donnees)) + bloc
                + struct.pack(">I", zlib.crc32(bloc) & 0xFFFFFFFF))

    entete = struct.pack(">IIBBBBB", largeur, hauteur, profondeur, 4, 0, 0, 0)
    with open(chemin, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(morceau(b"IHDR", entete))
        f.write(morceau(b"IDAT", zlib.compress(bytes(lignes), 6)))
        f.write(morceau(b"IEND", b""))


def image_de_test(largeur, hauteur, dtype=np.uint8, canaux=3, graine=7):
    """Image nettement rouge, texturee et contenant des zones sombres.

    Rouge, pour qu'une permutation de canaux se voie. Texturee, pour qu'un
    floutage change reellement les pixels. Sombre par endroits, pour qu'une
    normalisation entre -1 et 1 produise des valeurs negatives : une entree
    simplement divisee par 255 resterait dans [0, 1] et passerait inapercue.
    """
    maxi = 65535 if dtype == np.uint16 else 255
    rng = np.random.default_rng(graine)
    gy, gx = np.mgrid[0:hauteur, 0:largeur]
    rouge = 20 + (gx / max(1, largeur - 1)) * 180
    vert = 10 + (gy / max(1, hauteur - 1)) * 60
    bleu = 5 + ((gx + gy) % 32)
    texture = rng.integers(0, 24, size=(hauteur, largeur))
    couche = lambda base: np.clip(base + texture, 0, 255)
    rgb = np.dstack([couche(rouge), couche(vert), couche(bleu)])
    rgb = (rgb / 255.0 * maxi).astype(dtype)

    if canaux == 3:
        return rgb[:, :, ::-1].copy()                       # BGR pour OpenCV
    if canaux == 4:
        alpha = np.full((hauteur, largeur, 1), maxi, dtype=dtype)
        return np.dstack([rgb[:, :, ::-1], alpha])          # BGRA
    gris = (0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1]
            + 0.114 * rgb[:, :, 2]).astype(dtype)
    if canaux == 1:
        return gris
    alpha = np.full((hauteur, largeur), maxi, dtype=dtype)
    return np.dstack([gris, alpha])                         # gris + alpha


class Bac:
    """Dossier de travail isole, avec le double d'onnxruntime."""

    def __init__(self, scenario=None, avec_onnxruntime=True, avec_cv2=True):
        self.dossier = tempfile.mkdtemp(prefix="test_visage_")
        self.scenario = dict(scenario or {})
        self.faux = os.path.join(self.dossier, "faux_paquets")
        os.makedirs(self.faux, exist_ok=True)
        if avec_onnxruntime:
            with open(os.path.join(self.faux, "onnxruntime.py"), "w") as f:
                f.write(FAUX_ONNXRUNTIME)
        if not avec_cv2:
            with open(os.path.join(self.faux, "cv2.py"), "w") as f:
                f.write("raise ImportError('opencv absent (scenario de test)')\n")
        self.worker = os.path.join(self.dossier, "worker.py")
        with open(self.worker, "w", encoding="utf-8", newline="\n") as f:
            f.write(extraire_worker())
        self.modeles = {}
        for role, nom in MODELES_FACTICES.items():
            chemin = os.path.join(self.dossier, nom)
            with open(chemin, "wb") as f:
                f.write(b"\x08\x07onnx factice")
            self.modeles[role] = chemin
        self.trace = os.path.join(self.dossier, "trace.jsonl")

    def parametres(self, chemin_image, operations, taille_image, **extra):
        base = {
            "image": chemin_image,
            "dossier_sortie": self.dossier,
            "operations": list(operations),
            "modeles": dict(self.modeles),
            "fournisseurs": ["CPUExecutionProvider"],
            "roi_visages": None,
            "pivot_espace": extraire_constante("PIVOT_ESPACE"),
            "pivot_type": extraire_constante("PIVOT_TYPE"),
            "pause_entre_modeles_s": 0.0,
            "taille_entree_detection": extraire_constante("TAILLE_ENTREE_DETECTION"),
            "detection_normalisation": [0.0, 1.0],
            "detection_nombre_classes": 1,
            "score_min_visage": extraire_constante("SCORE_MIN_VISAGE"),
            "nms_recouvrement_max": extraire_constante("NMS_RECOUVREMENT_MAX"),
            "visages_max": extraire_constante("VISAGES_MAX"),
            "cote_min_visage_ratio": extraire_constante("COTE_MIN_VISAGE_RATIO"),
            "marge_visage_ratio": extraire_constante("MARGE_VISAGE_RATIO"),
            "marge_visage_min_px": extraire_constante("MARGE_VISAGE_MIN_PX"),
            "fusion_plume_ratio": extraire_constante("FUSION_PLUME_RATIO"),
            "anonymisation_mode": extraire_constante("ANONYME_FLOU"),
            "anonyme_flou_ratio": extraire_constante("ANONYME_FLOU_RATIO"),
            "anonyme_pixels_ratio": extraire_constante("ANONYME_PIXELS_RATIO"),
            "sourire_taille_entree": 256,
            "sourire_normalisation": [0.5, 0.5],
            "sourire_intensite": 1.0,
            "sourire_vecteur_taille": extraire_constante("SOURIRE_VECTEUR_TAILLE"),
            "sourire_indice_attribut": extraire_constante("SOURIRE_INDICE_ATTRIBUT"),
            "amelioration_taille_entree": 512,
            "amelioration_normalisation": [0.5, 0.5],
            "colorisation_taille_entree": 256,
            "colorisation_normalisation": [0.5, 0.5],
            "rehaussement_force": extraire_constante("REHAUSSEMENT_FORCE"),
            "rehaussement_rayon": extraire_constante("REHAUSSEMENT_RAYON"),
            "plafond_memoire_actif": False,
            "plafond_memoire_ratio": extraire_constante("MEMOIRE_PLAFOND_RATIO"),
            "plafond_memoire_octets": 0,
        }
        base.update(extra)
        verifier_couverture_des_cles(base)
        self.scenario["TEST_TAILLE_IMAGE"] = json.dumps(list(taille_image))
        return base

    def executer(self, parametres, arguments=None):
        chemin_cfg = os.path.join(self.dossier, "cfg.json")
        with open(chemin_cfg, "w", encoding="utf-8") as f:
            json.dump(parametres, f)
        env = dict(os.environ)
        env["PYTHONPATH"] = self.faux
        env["TEST_TRACE"] = self.trace
        env.update(self.scenario)
        commande = [sys.executable, self.worker]
        commande.extend(arguments if arguments is not None else [chemin_cfg])
        proc = subprocess.run(commande, capture_output=True, env=env, timeout=600)
        resultat = {}
        chemin_resultat = os.path.join(self.dossier, "resultat.json")
        if os.path.isfile(chemin_resultat):
            with open(chemin_resultat, encoding="utf-8") as f:
                resultat = json.load(f)
        sortie = os.path.join(self.dossier, "resultat.png")
        image = None
        if os.path.isfile(sortie) and os.path.getsize(sortie) > 0:
            image = cv2.imread(sortie, cv2.IMREAD_UNCHANGED)
        return {"code": proc.returncode, "resultat": resultat, "image": image,
                "journal": proc.stdout.decode("utf-8", "replace"),
                "erreurs": proc.stderr.decode("utf-8", "replace"),
                "evenements": self.evenements()}

    def evenements(self):
        if not os.path.isfile(self.trace):
            return []
        evenements = []
        with open(self.trace, encoding="utf-8") as f:
            for ligne in f:
                ligne = ligne.strip()
                if ligne:
                    evenements.append(json.loads(ligne))
        return evenements

    def nettoyer(self):
        shutil.rmtree(self.dossier, ignore_errors=True)


RESULTATS = []


def controler(nom, condition, detail=""):
    RESULTATS.append((nom, bool(condition), detail))
    marque = "OK   " if condition else "ECHEC"
    print("  %s %s%s" % (marque, nom,
                         (" - " + detail) if detail and not condition else ""))
    return bool(condition)


def rect_attendu(boite, largeur, hauteur):
    """Meme calcul que boite_avec_marge, refait ici a partir des constantes du
    greffon : c'est la reference contre laquelle la geometrie est verifiee."""
    ratio = extraire_constante("MARGE_VISAGE_RATIO")
    minimum = extraire_constante("MARGE_VISAGE_MIN_PX")
    x, y, bl, bh = boite
    marge = max(ratio * max(bl, bh), minimum)
    x1 = int(max(0, round(x - marge)))
    y1 = int(max(0, round(y - marge)))
    x2 = int(min(largeur, round(x + bl + marge)))
    y2 = int(min(hauteur, round(y + bh + marge)))
    return [x1, y1, x2 - x1, y2 - y1]


def canaux_couleur(image):
    if image.ndim == 2:
        return image[:, :, np.newaxis]
    if image.shape[2] in (2, 4):
        return image[:, :, :image.shape[2] - 1]
    return image


def test_geometrie_et_contrat():
    """Cas 1 : le masque atterrit-il sur le visage, et le modele recoit-il ce
    qu'il attend ?

    L'image n'est volontairement pas carree : c'est le remplissage de la mise
    en lettre-boite qui revele un oubli de conversion de coordonnees, et une
    image carree le masquerait entierement.
    """
    print("Cas 1 : geometrie des boites et contrat d'entree du detecteur")
    largeur, hauteur = 900, 600
    visages = [[120.0, 90.0, 96.0, 110.0, 0.92], [620.0, 330.0, 70.0, 84.0, 0.71]]
    bac = Bac({"TEST_VISAGES": json.dumps(visages)})
    try:
        source = os.path.join(bac.dossier, "source.png")
        cv2.imwrite(source, image_de_test(largeur, hauteur))
        origine = cv2.imread(source, cv2.IMREAD_UNCHANGED)
        sortie = bac.executer(bac.parametres(source, ["anonymiser"],
                                             (largeur, hauteur)))

        controler("cas 1 : le worker reussit", sortie["code"] == 0,
                  "code %s, %s" % (sortie["code"], sortie["journal"][-300:]))
        controler("cas 1 : une image est produite", sortie["image"] is not None)
        if sortie["image"] is None:
            return
        rects = [v["rect"] for v in sortie["resultat"].get("visages", [])]
        controler("cas 1 : les deux visages sont retenus", len(rects) == 2,
                  "%d retenu(s)" % len(rects))

        attendus = [rect_attendu(v[:4], largeur, hauteur) for v in visages]
        ecarts = []
        for attendu in attendus:
            proche = [r for r in rects
                      if max(abs(a - b) for a, b in zip(r, attendu)) <= 2]
            if not proche:
                ecarts.append("attendu %s, obtenus %s" % (attendu, rects))
        controler("cas 1 : chaque zone traitee tombe sur le visage designe "
                  "(tolerance 2 px)", not ecarts, " | ".join(ecarts))

        difference = np.abs(canaux_couleur(sortie["image"]).astype(np.int32)
                            - canaux_couleur(origine).astype(np.int32)).sum(axis=2)
        dehors = np.ones((hauteur, largeur), dtype=bool)
        for x, y, bl, bh in rects:
            dehors[y:y + bh, x:x + bl] = False
        controler("cas 1 : aucun pixel modifie hors des zones de visage",
                  int(difference[dehors].max() if dehors.any() else 0) == 0,
                  "ecart maximal hors zone : %d"
                  % int(difference[dehors].max() if dehors.any() else 0))
        interieurs = [int(difference[y + bh // 2, x + bl // 2])
                      for x, y, bl, bh in rects]
        controler("cas 1 : le centre de chaque visage est reellement modifie",
                  all(v > 0 for v in interieurs), str(interieurs))

        # Le fondu doit atteindre exactement zero sur l'anneau exterieur de la
        # zone : c'est ce qui rend la frontiere invisible et la promesse
        # "seuls les visages changent" verifiable au pixel pres.
        bords = []
        for x, y, bl, bh in rects:
            bords.append(int(difference[y, x:x + bl].max()))
            bords.append(int(difference[y + bh - 1, x:x + bl].max()))
            bords.append(int(difference[y:y + bh, x].max()))
            bords.append(int(difference[y:y + bh, x + bl - 1].max()))
        controler("cas 1 : le bord exterieur de chaque zone est exactement "
                  "inchange", max(bords) == 0 if bords else False, str(bords))

        sessions = [e for e in sortie["evenements"] if e["evenement"] == "session"]
        controler("cas 1 : une seule session de detection",
                  [e["role"] for e in sessions] == ["detection"],
                  str([e["role"] for e in sessions]))
        controler("cas 1 : les arenes memoire d'onnxruntime sont desactivees",
                  sessions and sessions[0]["mem_pattern"] is False
                  and sessions[0]["cpu_arena"] is False,
                  str(sessions[:1]))
    finally:
        bac.nettoyer()


def test_chainage_et_liberation():
    """Cas 2 : ordre des etapes, et liberation effective entre deux modeles."""
    print("Cas 2 : chainage colorisation -> detection -> sourire")
    largeur, hauteur = 640, 480
    visages = [[200.0, 150.0, 120.0, 140.0, 0.9]]
    bac = Bac({"TEST_VISAGES": json.dumps(visages)})
    try:
        source = os.path.join(bac.dossier, "source.png")
        cv2.imwrite(source, image_de_test(largeur, hauteur))
        sortie = bac.executer(bac.parametres(
            source, ["coloriser", "sourire"], (largeur, hauteur)))
        controler("cas 2 : le worker reussit", sortie["code"] == 0,
                  "code %s, %s" % (sortie["code"], sortie["journal"][-300:]))

        roles = [e["role"] for e in sortie["evenements"]
                 if e["evenement"] == "session"]
        controler("cas 2 : les modeles se chargent dans l'ordre du pipeline",
                  roles == ["colorisation", "detection", "sourire"], str(roles))

        # Un modele libere avant le chargement du suivant : entre deux
        # ouvertures, la fermeture de la precedente doit apparaitre.
        sequence = [(e["evenement"], e.get("role")) for e in sortie["evenements"]
                    if e["evenement"] in ("session", "fermeture")]
        ouvertes = 0
        maximum = 0
        for evenement, _ in sequence:
            ouvertes += 1 if evenement == "session" else -1
            maximum = max(maximum, ouvertes)
        controler("cas 2 : jamais plus d'un modele charge a la fois",
                  maximum <= 1, "maximum observe : %d, sequence %s"
                  % (maximum, sequence))

        entrees = [e for e in sortie["evenements"]
                   if e["evenement"] == "inference" and e["role"] == "sourire"]
        controler("cas 2 : le vecteur d'attributs est bien transmis",
                  entrees and len(entrees[0].get("entrees", [])) == 2,
                  str(entrees[:1]))

        # Une session ouverte ne prouve pas que l'etape a abouti : un contrat
        # d'entree viole ferait echouer le calcul, et la degradation
        # produirait tout de meme une image. C'est l'etape constatee dans le
        # rapport, et l'absence de degradation, qui font foi.
        etapes = [m[0] for m in (sortie["resultat"].get("moteurs") or [])]
        controler("cas 2 : le sourire est reellement applique",
                  "sourire" in etapes, str(etapes))
        controler("cas 2 : aucune etape n'a du degrader",
                  not (sortie["resultat"].get("degradations") or []),
                  str(sortie["resultat"].get("degradations")))
    finally:
        bac.nettoyer()


def test_seize_bits():
    """Cas 3 : profondeur 16 bits, ou une division par 255 blanchit l'image
    sans lever la moindre exception."""
    print("Cas 3 : image 16 bits")
    largeur, hauteur = 500, 400
    visages = [[150.0, 120.0, 90.0, 100.0, 0.88]]
    bac = Bac({"TEST_VISAGES": json.dumps(visages)})
    try:
        source = os.path.join(bac.dossier, "source.png")
        cv2.imwrite(source, image_de_test(largeur, hauteur, np.uint16, 3))
        origine = cv2.imread(source, cv2.IMREAD_UNCHANGED)
        controler("cas 3 : la source est bien relue en 16 bits",
                  origine is not None and origine.dtype == np.uint16,
                  str(getattr(origine, "dtype", None)))
        sortie = bac.executer(bac.parametres(source, ["anonymiser"],
                                             (largeur, hauteur)))
        controler("cas 3 : le worker reussit", sortie["code"] == 0,
                  "code %s, %s" % (sortie["code"], sortie["journal"][-300:]))
        image = sortie["image"]
        if image is None:
            return
        controler("cas 3 : la profondeur 16 bits est restituee",
                  image.dtype == np.uint16, str(image.dtype))
        controler("cas 3 : l'image n'a pas blanchi",
                  abs(float(image.mean()) - float(origine.mean()))
                  < 0.05 * 65535,
                  "moyennes %.0f -> %.0f" % (origine.mean(), image.mean()))
        rects = [v["rect"] for v in sortie["resultat"].get("visages", [])]
        dehors = np.ones((hauteur, largeur), dtype=bool)
        for x, y, bl, bh in rects:
            dehors[y:y + bh, x:x + bl] = False
        ecart = np.abs(image.astype(np.int64)
                       - origine.astype(np.int64)).sum(axis=2)
        controler("cas 3 : aucun pixel modifie hors des zones, en 16 bits",
                  int(ecart[dehors].max()) == 0,
                  "ecart maximal : %d" % int(ecart[dehors].max()))
        ecarts_zones = []
        for x, y, bl, bh in rects:
            avant = float(origine[y:y + bh, x:x + bl].mean())
            apres = float(image[y:y + bh, x:x + bl].mean())
            ecarts_zones.append(abs(apres - avant) / max(1.0, avant))
        controler("cas 3 : la zone traitee garde sa luminosite d'origine "
                  "(un 16 bits pris pour un 8 bits la saturerait)",
                  bool(ecarts_zones) and max(ecarts_zones) < 0.10,
                  str([round(v, 3) for v in ecarts_zones]))
    finally:
        bac.nettoyer()


def test_quatre_cas_de_canaux():
    """Cas 4 : gris, gris + alpha, couleur, couleur + alpha."""
    print("Cas 4 : les quatre combinaisons de canaux")
    largeur, hauteur = 420, 360
    visages = [[140.0, 110.0, 90.0, 100.0, 0.9]]
    for canaux, libelle in ((1, "gris"), (2, "gris + alpha"),
                            (3, "couleur"), (4, "couleur + alpha")):
        bac = Bac({"TEST_VISAGES": json.dumps(visages),
                   "TEST_SOURCE_GRISE": "1" if canaux in (1, 2) else "0"})
        try:
            source = os.path.join(bac.dossier, "source.png")
            tableau = image_de_test(largeur, hauteur, np.uint8, canaux)
            if canaux == 2:
                png_gris_alpha(source, tableau)
            else:
                cv2.imwrite(source, tableau)
            relu = cv2.imread(source, cv2.IMREAD_UNCHANGED)
            attendus = {1: 2, 2: 3, 3: 3, 4: 3}
            controler("cas 4 (%s) : la source se relit" % libelle,
                      relu is not None and relu.ndim == attendus[canaux],
                      "ndim %s" % (getattr(relu, "ndim", None),))
            sortie = bac.executer(bac.parametres(source, ["anonymiser"],
                                                 (largeur, hauteur)))
            controler("cas 4 (%s) : le worker reussit" % libelle,
                      sortie["code"] == 0,
                      "code %s, %s" % (sortie["code"], sortie["journal"][-300:]))
            image = sortie["image"]
            if image is None:
                continue
            if canaux == 1:
                controler("cas 4 (gris) : la sortie reste en niveaux de gris",
                          image.ndim == 2, "forme %s" % (image.shape,))
            elif canaux == 2:
                controler("cas 4 (gris + alpha) : la transparence est conservee",
                          image.ndim == 3 and image.shape[2] == 4,
                          "forme %s" % (image.shape,))
            elif canaux == 4:
                controler("cas 4 (couleur + alpha) : la transparence est "
                          "conservee",
                          image.ndim == 3 and image.shape[2] == 4,
                          "forme %s" % (image.shape,))
            else:
                controler("cas 4 (couleur) : trois canaux en sortie",
                          image.ndim == 3 and image.shape[2] == 3,
                          "forme %s" % (image.shape,))
        finally:
            bac.nettoyer()


def test_colorisation_conserve_la_luminance():
    """Cas 5 : la colorisation ajoute de la couleur sans deplacer la
    luminance. C'est la propriete verifiable qui distingue un resultat colorise
    d'une image simplement eclaircie ou assombrie."""
    print("Cas 5 : colorisation, luminance conservee et chrominance ajoutee")
    largeur, hauteur = 320, 240
    # Teinte moderee, du domaine des carnations : une couleur saturee sort du
    # gamut a la reconversion et deplace la luminance pour une raison qui n'a
    # rien a voir avec le greffon.
    bac = Bac({"TEST_VISAGES": "[]", "TEST_SOURCE_GRISE": "1",
               "TEST_TEINTE": json.dumps([0.62, 0.50, 0.42])})
    try:
        source = os.path.join(bac.dossier, "source.png")
        cv2.imwrite(source, image_de_test(largeur, hauteur, np.uint8, 1))
        origine = cv2.imread(source, cv2.IMREAD_UNCHANGED)
        sortie = bac.executer(bac.parametres(source, ["coloriser"],
                                             (largeur, hauteur)))
        controler("cas 5 : le worker reussit", sortie["code"] == 0,
                  "code %s, %s" % (sortie["code"], sortie["journal"][-300:]))
        image = sortie["image"]
        if image is None:
            return
        controler("cas 5 : une image grise devient une image couleur",
                  image.ndim == 3 and image.shape[2] == 3, str(image.shape))
        lab_origine = cv2.cvtColor(cv2.cvtColor(origine, cv2.COLOR_GRAY2BGR),
                                   cv2.COLOR_BGR2LAB)
        lab_sortie = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        ecart_l = float(np.abs(lab_sortie[:, :, 0].astype(np.int32)
                               - lab_origine[:, :, 0].astype(np.int32)).max())
        controler("cas 5 : la luminance reste celle de l'original",
                  ecart_l <= 4.0, "ecart maximal sur L : %.0f" % ecart_l)
        chroma = float(np.abs(lab_sortie[:, :, 1].astype(np.int32) - 128).mean())
        controler("cas 5 : de la chrominance a bien ete ajoutee", chroma > 5.0,
                  "ecart moyen du canal a : %.1f" % chroma)
    finally:
        bac.nettoyer()


def test_preseance_cote_worker():
    """Cas 6 : la table de preseance du worker, second filet.

    Le greffon expurge normalement les conflits avant d'ecrire le fichier de
    parametres. Ce test alimente le worker avec un conflit pour verifier qu'il
    le rattrape, et surtout qu'il le SIGNALE - c'est ce drapeau qui rend le
    mecanisme du greffon testable separement.
    """
    print("Cas 6 : preseance de l'anonymisation, cote worker")
    largeur, hauteur = 400, 320
    visages = [[120.0, 100.0, 90.0, 100.0, 0.9]]
    bac = Bac({"TEST_VISAGES": json.dumps(visages)})
    try:
        source = os.path.join(bac.dossier, "source.png")
        cv2.imwrite(source, image_de_test(largeur, hauteur))
        sortie = bac.executer(bac.parametres(
            source, ["anonymiser", "sourire", "ameliorer"], (largeur, hauteur)))
        controler("cas 6 : le worker reussit", sortie["code"] == 0,
                  "code %s" % sortie["code"])
        resultat = sortie["resultat"]
        controler("cas 6 : le conflit est rattrape et signale",
                  resultat.get("preseance_appliquee_par_worker") is True,
                  str(resultat.get("preseance_appliquee_par_worker")))
        controler("cas 6 : seule l'anonymisation subsiste",
                  resultat.get("operations") == ["anonymiser"],
                  str(resultat.get("operations")))
        roles = [e["role"] for e in sortie["evenements"]
                 if e["evenement"] == "session"]
        controler("cas 6 : aucun modele de visage n'a ete charge",
                  roles == ["detection"], str(roles))
    finally:
        bac.nettoyer()


def test_degradation_sans_modele():
    """Cas 7 : un modele absent degrade, il n'interrompt pas.

    L'amelioration dispose d'un chemin sans IA ; le sourire non. Les deux cas
    doivent produire une image et dire ce qui s'est passe.
    """
    print("Cas 7 : degradation quand un modele manque")
    largeur, hauteur = 400, 320
    visages = [[120.0, 100.0, 90.0, 100.0, 0.9]]
    bac = Bac({"TEST_VISAGES": json.dumps(visages)})
    try:
        source = os.path.join(bac.dossier, "source.png")
        cv2.imwrite(source, image_de_test(largeur, hauteur))
        parametres = bac.parametres(source, ["sourire", "ameliorer"],
                                    (largeur, hauteur))
        parametres["modeles"] = dict(parametres["modeles"])
        parametres["modeles"]["sourire"] = None
        parametres["modeles"]["amelioration"] = None
        sortie = bac.executer(parametres)
        controler("cas 7 : une image est tout de meme produite",
                  sortie["code"] == 0 and sortie["image"] is not None,
                  "code %s" % sortie["code"])
        degradations = sortie["resultat"].get("degradations") or []
        controler("cas 7 : le sourire ignore est explique",
                  any("sourire" in d for d in degradations), str(degradations))
        moteurs = [m[1] for m in (sortie["resultat"].get("moteurs") or [])]
        controler("cas 7 : l'amelioration passe par son repli sans IA",
                  any("sans IA" in m for m in moteurs), str(moteurs))
    finally:
        bac.nettoyer()


def test_visages_a_l_extremite_difficile():
    """Cas 8 : le jeu d'essai est dimensionne sur le cas defavorable annonce.

    Le seuil de taille est une hypothese sur la taille des sujets. Un jeu
    d'essai confortable ne la confronte jamais : on prend donc un visage juste
    au-dessus du seuil et un juste en dessous, aux dimensions reelles que la
    constante pretend couvrir.
    """
    print("Cas 8 : visages a la limite du seuil de taille")
    largeur, hauteur = 900, 600
    ratio = extraire_constante("COTE_MIN_VISAGE_RATIO")
    cote_seuil = ratio * min(largeur, hauteur)
    garde = round(cote_seuil * 1.2, 1)
    ecarte = round(cote_seuil * 0.5, 1)
    visages = [[300.0, 200.0, garde, garde, 0.9],
               [600.0, 400.0, ecarte, ecarte, 0.9]]
    bac = Bac({"TEST_VISAGES": json.dumps(visages)})
    try:
        source = os.path.join(bac.dossier, "source.png")
        cv2.imwrite(source, image_de_test(largeur, hauteur))
        sortie = bac.executer(bac.parametres(source, ["anonymiser"],
                                             (largeur, hauteur)))
        rects = sortie["resultat"].get("visages", [])
        controler("cas 8 : le visage au-dessus du seuil (%.1f px) est traite, "
                  "celui en dessous (%.1f px) est ecarte" % (garde, ecarte),
                  len(rects) == 1, "%d zone(s) : %s" % (len(rects), rects))
    finally:
        bac.nettoyer()


def test_doublons_de_detection():
    """Cas 9 : deux boites qui se recouvrent largement designent un seul
    visage."""
    print("Cas 9 : suppression des doublons de detection")
    largeur, hauteur = 640, 480
    visages = [[200.0, 150.0, 120.0, 140.0, 0.95],
               [206.0, 156.0, 120.0, 140.0, 0.80],
               [450.0, 100.0, 100.0, 120.0, 0.85]]
    bac = Bac({"TEST_VISAGES": json.dumps(visages)})
    try:
        source = os.path.join(bac.dossier, "source.png")
        cv2.imwrite(source, image_de_test(largeur, hauteur))
        sortie = bac.executer(bac.parametres(source, ["anonymiser"],
                                             (largeur, hauteur)))
        rects = sortie["resultat"].get("visages", [])
        controler("cas 9 : les deux boites superposees n'en font qu'une",
                  len(rects) == 2, "%d zone(s)" % len(rects))
    finally:
        bac.nettoyer()


def test_sorties_en_echec():
    """Cas 10 : aucun chemin de sortie sans marqueur ni fichier temoin.

    Un sys.exit(1) nu produirait chez l'utilisateur un message au contenu vide.
    Chaque echec porte ici son marqueur, son code et son temoin sur disque.
    """
    print("Cas 10 : marqueur et fichier temoin pour chaque sortie en echec")
    largeur, hauteur = 320, 240

    bac = Bac({"TEST_VISAGES": "[]"})
    try:
        sortie = bac.executer(bac.parametres(
            os.path.join(bac.dossier, "inexistant.png"), ["anonymiser"],
            (largeur, hauteur)), arguments=[])
        controler("cas 10 : aucun parametre -> [ERR_PARAMS]",
                  sortie["code"] == 2 and "[ERR_PARAMS]" in sortie["journal"],
                  "code %s" % sortie["code"])
    finally:
        bac.nettoyer()

    bac = Bac({"TEST_VISAGES": "[]"})
    try:
        sortie = bac.executer(bac.parametres(
            os.path.join(bac.dossier, "inexistant.png"), ["anonymiser"],
            (largeur, hauteur)))
        controler("cas 10 : image illisible -> [ERR_IMAGE]",
                  sortie["code"] == 4
                  and sortie["resultat"].get("marqueur") == "[ERR_IMAGE]",
                  "code %s, %s" % (sortie["code"], sortie["resultat"]))
    finally:
        bac.nettoyer()

    bac = Bac({"TEST_VISAGES": "[]", "TEST_ECHEC": "modele:colorisation"})
    try:
        source = os.path.join(bac.dossier, "source.png")
        cv2.imwrite(source, image_de_test(largeur, hauteur))
        sortie = bac.executer(bac.parametres(source, ["coloriser"],
                                             (largeur, hauteur)))
        controler("cas 10 : modele illisible -> [ERR_MODELE]",
                  sortie["code"] == 5
                  and sortie["resultat"].get("marqueur") == "[ERR_MODELE]",
                  "code %s, %s" % (sortie["code"],
                                   sortie["resultat"].get("marqueur")))
    finally:
        bac.nettoyer()

    bac = Bac({"TEST_VISAGES": "[]", "TEST_ECHEC": "inference:colorisation"})
    try:
        source = os.path.join(bac.dossier, "source.png")
        cv2.imwrite(source, image_de_test(largeur, hauteur))
        sortie = bac.executer(bac.parametres(source, ["coloriser"],
                                             (largeur, hauteur)))
        controler("cas 10 : inference en echec -> [ERR_INFERENCE]",
                  sortie["code"] == 7
                  and sortie["resultat"].get("marqueur") == "[ERR_INFERENCE]",
                  "code %s, %s" % (sortie["code"],
                                   sortie["resultat"].get("marqueur")))
    finally:
        bac.nettoyer()

    bac = Bac({"TEST_VISAGES": "[]"})
    try:
        source = os.path.join(bac.dossier, "source.png")
        cv2.imwrite(source, image_de_test(largeur, hauteur))
        sortie = bac.executer(bac.parametres(source, ["anonymiser"],
                                             (largeur, hauteur)))
        controler("cas 10 : aucun visage -> [ERR_AUCUN_VISAGE]",
                  sortie["code"] == 8
                  and sortie["resultat"].get("marqueur") == "[ERR_AUCUN_VISAGE]",
                  "code %s, %s" % (sortie["code"],
                                   sortie["resultat"].get("marqueur")))
    finally:
        bac.nettoyer()

    bac = Bac({"TEST_VISAGES": "[]", "TEST_SORTIE_ABERRANTE": "colorisation"})
    try:
        source = os.path.join(bac.dossier, "source.png")
        cv2.imwrite(source, image_de_test(largeur, hauteur))
        sortie = bac.executer(bac.parametres(source, ["coloriser"],
                                             (largeur, hauteur)))
        controler("cas 10 : sortie de modele aberrante -> [ERR_PIVOT]",
                  sortie["code"] == 6
                  and sortie["resultat"].get("marqueur") == "[ERR_PIVOT]",
                  "code %s, %s" % (sortie["code"],
                                   sortie["resultat"].get("marqueur")))
    finally:
        bac.nettoyer()

    bac = Bac({"TEST_VISAGES": "[]"}, avec_cv2=False)
    try:
        source = os.path.join(bac.dossier, "source.png")
        cv2.imwrite(source, image_de_test(largeur, hauteur))
        sortie = bac.executer(bac.parametres(source, ["anonymiser"],
                                             (largeur, hauteur)))
        controler("cas 10 : dependance absente -> [ERR_IMPORT]",
                  sortie["code"] == 3
                  and sortie["resultat"].get("marqueur") == "[ERR_IMPORT]",
                  "code %s, %s" % (sortie["code"],
                                   sortie["resultat"].get("marqueur")))
    finally:
        bac.nettoyer()


PROBE_MEMOIRE = r'''
import ast
import sys

source = open(sys.argv[1], encoding="utf-8").read()
espace = {"__name__": "worker_sous_test"}
exec(compile(source, "worker", "exec"), espace)

limite = int(sys.argv[2])
notes = []
pose = espace["poser_plafond_memoire"](
    {"plafond_memoire_actif": True, "plafond_memoire_octets": limite,
     "plafond_memoire_ratio": 0.75}, notes)
if pose <= 0:
    sys.stderr.write("plafond non pose: %s" % notes)
    sys.exit(3)
try:
    bloc = bytearray(limite * 6)
except MemoryError:
    sys.exit(0)
sys.stderr.write("allocation de %d octets acceptee malgre le plafond"
                 % (limite * 6))
sys.exit(4)
'''


def test_plafond_memoire():
    """Cas 11 : la protection memoire se prouve, elle ne se suppose pas.

    Une protection inverifiable se teste, elle ne se supprime pas d'office : on
    pose une limite basse, on demande au processus d'allouer davantage, et le
    resultat se lit dans le code de sortie. Sur une version ou le setrlimit
    serait retire, l'allocation reussit et ce test passe au rouge.
    """
    print("Cas 11 : plafond memoire (POSIX)")
    if os.name == "nt":
        controler("cas 11 : plafond memoire non pose sous Windows, et "
                  "documente comme tel", True)
        return
    dossier = tempfile.mkdtemp(prefix="test_plafond_")
    try:
        worker = os.path.join(dossier, "worker.py")
        with open(worker, "w", encoding="utf-8", newline="\n") as f:
            f.write(extraire_worker())
        sonde = os.path.join(dossier, "sonde.py")
        with open(sonde, "w", encoding="utf-8", newline="\n") as f:
            f.write(PROBE_MEMOIRE)
        limite = 200 * 1024 * 1024
        proc = subprocess.run([sys.executable, sonde, worker, str(limite)],
                              capture_output=True, timeout=120)
        controler("cas 11 : une allocation au-dela du plafond est refusee par "
                  "le systeme", proc.returncode == 0,
                  "code %s, %s" % (proc.returncode,
                                   proc.stderr.decode("utf-8", "replace")[:200]))
    finally:
        shutil.rmtree(dossier, ignore_errors=True)


def main():
    print("Tests du worker IA Visage Studio "
          "(double d'onnxruntime, aucun modele reel)")
    print("")
    test_geometrie_et_contrat()
    test_chainage_et_liberation()
    test_seize_bits()
    test_quatre_cas_de_canaux()
    test_colorisation_conserve_la_luminance()
    test_preseance_cote_worker()
    test_degradation_sans_modele()
    test_visages_a_l_extremite_difficile()
    test_doublons_de_detection()
    test_sorties_en_echec()
    test_plafond_memoire()
    print("")
    echecs = [nom for nom, ok, _ in RESULTATS if not ok]
    print("%d controles, %d echec(s)" % (len(RESULTATS), len(echecs)))
    for nom in echecs:
        print("  ECHEC : " + nom)
    return 1 if echecs else 0


if __name__ == "__main__":
    sys.exit(main())
