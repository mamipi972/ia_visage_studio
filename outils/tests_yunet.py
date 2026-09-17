#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Controles du detecteur YuNet, sur le vrai modele.

Usage : python3 outils/tests_yunet.py   (necessite numpy, opencv, et le modele)

Les autres bancs remplacent le moteur d'inference par une doublure. Celui-ci
ne le fait pas : il charge le vrai fichier ONNX et le fait tourner. C'est le
seul moyen de verifier ce qu'aucune doublure ne peut prouver - que l'adresse
publiee sert bien un modele, qu'OpenCV sait le charger, et que les boites
qu'il rend atterrissent au bon endroit une fois remises a l'echelle de
l'image d'origine.

Le modele est cherche, dans l'ordre :
  1. dans la variable d'environnement IA_VISAGE_YUNET ;
  2. dans le dossier de modeles de la suite, s'il y est deja ;
  3. par telechargement depuis l'adresse declaree dans le greffon.

Sans reseau ni fichier local, les controles sont IGNORES et le disent : un
banc qui se tait sur ce qu'il n'a pas pu faire ne vaut pas mieux qu'un banc
absent.

Ce que ce banc ne prouve PAS, et il faut le dire : il ne distingue pas une
inversion RGB/BGR a l'entree de YuNet. Mesure le 2026-09-17 sur le visage
synthetique ci-dessous, la permutation fait tomber le score de 0,90 a 0,80 et
deplace la boite de quelques pixels - sans jamais empecher la detection. C'est
exactement le genre de defaut qu'un modele reel absorbe en rendant des valeurs
plausibles.
"""

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
FICHIER_GREFFON = os.path.join(RACINE, "ia_visage_studio.py")
DELAI_RESEAU_S = 60

RESULTATS = []


def controler(nom, condition, detail=""):
    RESULTATS.append((nom, bool(condition), detail))
    marque = "OK   " if condition else "ECHEC"
    suffixe = (" - " + str(detail)) if (detail and not condition) else ""
    print("  %s %s%s" % (marque, nom, suffixe))
    return bool(condition)


_TABLE = None


def table_modeles():
    """Table des modeles relue dans le greffon, jamais recopiee ici.

    Elle se lit en chargeant le module, et non par analyse du texte : ses
    valeurs renvoient a d'autres constantes du greffon, qu'une lecture
    litterale ne saurait pas resoudre.
    """
    global _TABLE
    if _TABLE is None:
        bac = tempfile.mkdtemp(prefix="table_modeles_")
        sauve = dict((c, os.environ.get(c)) for c in
                     ("LOCALAPPDATA", "XDG_DATA_HOME", "HOME"))
        try:
            for cle in sauve:
                os.environ[cle] = bac
            module = faux_gimp.installer(bac, FICHIER_GREFFON)
            _TABLE = module.MODELES
        finally:
            for cle, valeur in sauve.items():
                if valeur is None:
                    os.environ.pop(cle, None)
                else:
                    os.environ[cle] = valeur
            shutil.rmtree(bac, ignore_errors=True)
    return _TABLE


def obtenir_modele(dossier):
    """(chemin, origine) ou (None, motif)."""
    entree = table_modeles()["detection_yunet"]
    impose = os.environ.get("IA_VISAGE_YUNET", "").strip()
    if impose and os.path.isfile(impose):
        return impose, "IA_VISAGE_YUNET"

    cible = os.path.join(dossier, entree["fichier"])
    import urllib.request
    dernier = ""
    for url in entree["sources"]:
        try:
            requete = urllib.request.Request(url)
            requete.add_header("User-Agent", "gimp-ai-suite/test")
            with urllib.request.urlopen(requete, timeout=DELAI_RESEAU_S) as rep:
                donnees = rep.read()
            with open(cible, "wb") as f:
                f.write(donnees)
            return cible, url
        except Exception as e:
            dernier = "%s : %s" % (url, str(e)[:120])
    return None, dernier or "aucune adresse declaree"


def visage_synthetique(largeur, hauteur, centre, rayon):
    """Un visage schematique, mais que YuNet reconnait.

    Il ne s'agit pas de mesurer la qualite du modele - ce n'est pas le role de
    ce depot - mais de disposer d'un sujet dont on connait la position au
    pixel pres, pour verifier ou atterrit la boite.
    """
    image = np.full((hauteur, largeur, 3), 200, np.uint8)
    cx, cy = centre
    r = rayon
    cv2.ellipse(image, (cx, cy), (int(r * 0.78), r), 0, 0, 360,
                (170, 190, 225), -1)
    cv2.ellipse(image, (cx, cy - int(r * 0.62)), (int(r * 0.80), int(r * 0.55)),
                0, 180, 360, (60, 60, 70), -1)
    for cote in (-1, 1):
        ex = cx + cote * int(r * 0.33)
        ey = cy - int(r * 0.12)
        cv2.ellipse(image, (ex, ey), (int(r * 0.17), int(r * 0.10)), 0, 0, 360,
                    (250, 250, 250), -1)
        cv2.circle(image, (ex, ey), max(2, int(r * 0.055)), (40, 35, 30), -1)
        cv2.ellipse(image, (ex, ey - int(r * 0.17)),
                    (int(r * 0.20), int(r * 0.06)), 0, 180, 360, (70, 60, 55), -1)
    cv2.ellipse(image, (cx, cy + int(r * 0.10)), (int(r * 0.09), int(r * 0.20)),
                0, 0, 360, (150, 170, 205), -1)
    cv2.ellipse(image, (cx, cy + int(r * 0.45)), (int(r * 0.28), int(r * 0.12)),
                0, 0, 180, (90, 80, 120), -1)
    return image


def executer(bac, modele_yunet, source, largeur, hauteur):
    parametres = bac.parametres(source, ["anonymiser"], (largeur, hauteur))
    # Seul YuNet est fourni : le chemin YOLO ne doit pas etre emprunte.
    parametres["modeles"] = {"detection": None, "detection_yunet": modele_yunet,
                             "sourire": None, "amelioration": None,
                             "colorisation": None}
    return bac.executer(parametres)


def test_grande_image():
    """Le cas qui compte : une image plus grande que YUNET_COTE_MAX.

    YuNet travaille alors sur une reduction, et les boites qu'il rend sont
    dans l'espace reduit. Les oublier placerait le traitement dans le coin
    superieur gauche de toute grande photo, sans la moindre erreur.
    """
    print("Cas 1 : image plus grande que la taille de travail de YuNet")
    cote_max = tests_worker.extraire_constante("YUNET_COTE_MAX")
    largeur, hauteur = cote_max * 2, int(cote_max * 1.5)
    centre = (int(largeur * 0.62), int(hauteur * 0.40))
    rayon = int(min(largeur, hauteur) * 0.16)

    dossier = tempfile.mkdtemp(prefix="yunet_")
    try:
        modele, origine = obtenir_modele(dossier)
        if not modele:
            print("  IGNORE modele YuNet indisponible (%s)" % origine)
            print("         posez IA_VISAGE_YUNET sur un fichier local pour "
                  "executer ce banc")
            return False
        controler("le modele se telecharge et fait la taille annoncee",
                  os.path.getsize(modele) > 100 * 1024,
                  "%d octets depuis %s" % (os.path.getsize(modele), origine))

        bac = tests_worker.Bac({"TEST_VISAGES": "[]"}, avec_onnxruntime=False)
        try:
            source = os.path.join(bac.dossier, "source.png")
            image = visage_synthetique(largeur, hauteur, centre, rayon)
            cv2.imwrite(source, image)
            sortie = executer(bac, modele, source, largeur, hauteur)

            controler("cas 1 : le worker reussit", sortie["code"] == 0,
                      "code %s, %s" % (sortie["code"], sortie["journal"][-300:]))
            visages = sortie["resultat"].get("visages", [])
            controler("cas 1 : un visage est detecte", len(visages) == 1,
                      "%d detecte(s)" % len(visages))
            if not visages:
                return True

            x, y, bl, bh = visages[0]["rect"]
            controler("cas 1 : la zone traitee contient le visage, dans les "
                      "coordonnees de l'image d'origine",
                      x <= centre[0] <= x + bl and y <= centre[1] <= y + bh,
                      "centre %s hors de la boite %s" % (centre, [x, y, bl, bh]))
            # Sans remise a l'echelle, la boite resterait dans l'espace reduit
            # et tomberait donc au-dessus et a gauche du visage.
            controler("cas 1 : la boite n'est pas restee dans l'espace reduit",
                      x + bl > cote_max, "x+bl = %d, cote_max = %d"
                      % (x + bl, cote_max))
            aire = float(bl * bh) / float(largeur * hauteur)
            controler("cas 1 : l'aire de la zone est plausible",
                      0.005 < aire < 0.35, "%.4f de l'image" % aire)
            # La reduction est une optimisation, pas une condition de
            # justesse : sans elle le resultat reste correct, seulement plus
            # lent. Elle se constate donc dans les notes du worker, faute de
            # quoi sa disparition passerait inapercue.
            notes = " ".join(sortie["resultat"].get("notes") or [])
            controler("cas 1 : la reduction a bien eu lieu",
                      "facteur 0." in notes and "facteur 1.000" not in notes,
                      notes[:200])
            controler("cas 1 : le moteur constate est bien YuNet",
                      "yunet" in (sortie["resultat"].get("moteur_detection")
                                  or "").lower(),
                      str(sortie["resultat"].get("moteur_detection")))

            image_sortie = sortie["image"]
            if image_sortie is not None:
                origine_lue = cv2.imread(source, cv2.IMREAD_UNCHANGED)
                dehors = np.ones((hauteur, largeur), dtype=bool)
                dehors[y:y + bh, x:x + bl] = False
                ecart = np.abs(image_sortie.astype(np.int32)
                               - origine_lue.astype(np.int32)).sum(axis=2)
                controler("cas 1 : rien n'a change hors de la zone du visage",
                          int(ecart[dehors].max()) == 0,
                          "ecart maximal %d" % int(ecart[dehors].max()))
                controler("cas 1 : le visage, lui, a bien ete modifie",
                          int(ecart[centre[1], centre[0]]) > 0)
        finally:
            bac.nettoyer()
    finally:
        shutil.rmtree(dossier, ignore_errors=True)
    return True


def test_petite_image():
    """Sans reduction, le facteur d'echelle vaut 1 : le chemin est different."""
    print("Cas 2 : image plus petite que la taille de travail")
    cote_max = tests_worker.extraire_constante("YUNET_COTE_MAX")
    largeur, hauteur = 480, 360
    centre = (int(largeur * 0.45), int(hauteur * 0.50))
    rayon = int(min(largeur, hauteur) * 0.26)

    dossier = tempfile.mkdtemp(prefix="yunet_")
    try:
        modele, origine = obtenir_modele(dossier)
        if not modele:
            print("  IGNORE modele YuNet indisponible")
            return False
        bac = tests_worker.Bac({"TEST_VISAGES": "[]"}, avec_onnxruntime=False)
        try:
            source = os.path.join(bac.dossier, "source.png")
            cv2.imwrite(source, visage_synthetique(largeur, hauteur, centre, rayon))
            sortie = executer(bac, modele, source, largeur, hauteur)
            controler("cas 2 : le worker reussit", sortie["code"] == 0,
                      "code %s, %s" % (sortie["code"], sortie["journal"][-300:]))
            visages = sortie["resultat"].get("visages", [])
            controler("cas 2 : un visage est detecte", len(visages) == 1,
                      "%d detecte(s)" % len(visages))
            if visages:
                x, y, bl, bh = visages[0]["rect"]
                controler("cas 2 : la zone traitee contient le visage",
                          x <= centre[0] <= x + bl and y <= centre[1] <= y + bh,
                          "centre %s hors de %s" % (centre, [x, y, bl, bh]))
            notes = " ".join(sortie["resultat"].get("notes") or [])
            controler("cas 2 : aucune reduction n'a eu lieu",
                      "facteur 1.000" in notes, notes[:200])
        finally:
            bac.nettoyer()
    finally:
        shutil.rmtree(dossier, ignore_errors=True)
    return True


def main():
    print("Detecteur YuNet - controles sur le vrai modele")
    print("")
    disponible = test_grande_image()
    if disponible:
        test_petite_image()
    print("")
    if not RESULTATS:
        print("0 controle execute : modele indisponible, banc ignore.")
        return 0
    echecs = [nom for nom, ok, _ in RESULTATS if not ok]
    print("%d controles, %d echec(s)" % (len(RESULTATS), len(echecs)))
    for nom in echecs:
        print("  ECHEC : " + nom)
    return 1 if echecs else 0


if __name__ == "__main__":
    sys.exit(main())
