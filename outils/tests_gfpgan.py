#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Controle du modele d'amelioration, sur le vrai fichier.

Usage : python3 outils/tests_gfpgan.py

Aucune adresse publique ne sert gfpgan_1_4.onnx : les trois essayees
renvoyaient 401 ou 404. Le fichier se fabrique donc localement, par
outils/conversion/convertir_gfpgan.py. Ce banc verifie ce que la conversion
seule ne peut pas dire : que le greffon, avec son propre code d'inference et
sans aucune doublure, charge ce fichier, l'emploie, et n'en tire ni erreur ni
degradation.

Le modele est cherche, dans l'ordre :
  1. dans la variable d'environnement IA_VISAGE_GFPGAN ;
  2. dans le dossier de modeles de la suite.

Absent, les controles sont IGNORES et le disent. Ils ne sont pas telecharges
faute d'adresse, et ils ne sont surtout pas remplaces par une doublure : une
doublure prouverait que le worker sait appeler onnxruntime, ce que d'autres
bancs etablissent deja, et ne prouverait rien du fichier.

Ce que ce banc ne prouve PAS : que le resultat soit beau. GFPGAN reconstruit
un visage, et sur un sujet synthetique le rendu n'a aucun sens esthetique.
Seul est verifie que le traitement a lieu, a l'endroit prevu, et nulle part
ailleurs.
"""

import json
import os
import sys

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tests_worker
import tests_yunet

RESULTATS = []

NOM_GFPGAN = "gfpgan_1_4.onnx"
NOM_YUNET = "face_detection_yunet_2023mar.onnx"


def controler(nom, condition, detail=""):
    RESULTATS.append((nom, bool(condition), detail))
    marque = "OK   " if condition else "ECHEC"
    suffixe = (" - " + str(detail)) if (detail and not condition) else ""
    print("  %s %s%s" % (marque, nom, suffixe))
    return bool(condition)


def dossier_modeles_suite():
    """Emplacement canonique des modeles de la suite, hors profil GIMP."""
    force = os.environ.get("GIMP_AI_SUITE_DIR", "").strip()
    if force:
        return os.path.join(force, "models")
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA",
                              os.path.expanduser("~\\AppData\\Local"))
    elif sys.platform == "darwin":
        base = os.path.expanduser("~/Library/Application Support")
    else:
        base = os.environ.get("XDG_DATA_HOME",
                              os.path.expanduser("~/.local/share"))
    return os.path.join(base, "GIMP", "ai_suite_shared", "models")


def chercher(variable, nom_fichier):
    impose = os.environ.get(variable, "").strip()
    if impose:
        return (impose, "variable " + variable) if os.path.isfile(impose) else (None, "variable " + variable + " : fichier absent")
    candidat = os.path.join(dossier_modeles_suite(), nom_fichier)
    if os.path.isfile(candidat):
        return candidat, "dossier de la suite"
    return None, "absent du dossier de la suite"


def test_le_greffon_emploie_le_modele():
    """Le seul controle qui compte : le worker livre, le vrai onnxruntime, et
    les deux vrais fichiers."""
    print("Amelioration des visages, sur le vrai modele")

    gfpgan, ou_gfpgan = chercher("IA_VISAGE_GFPGAN", NOM_GFPGAN)
    yunet, ou_yunet = chercher("IA_VISAGE_YUNET", NOM_YUNET)
    if not gfpgan or not yunet:
        manquant = NOM_GFPGAN if not gfpgan else NOM_YUNET
        raison = ou_gfpgan if not gfpgan else ou_yunet
        print("  IGNORE %s introuvable (%s)" % (manquant, raison))
        if not gfpgan:
            print("         fabriquez-le : python3 outils/conversion/"
                  "convertir_gfpgan.py")
        if not yunet:
            print("         le banc outils/tests_yunet.py sait le telecharger")
        return False

    controler("le modele fait une taille plausible",
              os.path.getsize(gfpgan) > 100 * 1024 * 1024,
              "%d octets" % os.path.getsize(gfpgan))

    # Geometrie du banc YuNet : le seul sujet dont on sache qu'il est detecte.
    cote = tests_worker.extraire_constante("YUNET_COTE_MAX")
    largeur, hauteur = cote * 2, int(cote * 1.5)
    centre = (int(largeur * 0.62), int(hauteur * 0.40))
    rayon = int(min(largeur, hauteur) * 0.16)

    bac = tests_worker.Bac(avec_onnxruntime=False)   # le vrai onnxruntime
    try:
        source = os.path.join(bac.dossier, "entree.png")
        cv2.imwrite(source, tests_yunet.visage_synthetique(largeur, hauteur,
                                                           centre, rayon))
        avant = cv2.imread(source, cv2.IMREAD_UNCHANGED)

        parametres = bac.parametres(source, ["ameliorer"], (largeur, hauteur))
        parametres["modeles"] = {"detection": None, "detection_yunet": yunet,
                                 "sourire": None, "amelioration": gfpgan,
                                 "colorisation": None}
        parametres["fournisseurs"] = ["CPUExecutionProvider"]
        bac.scenario.pop("TEST_TAILLE_IMAGE", None)
        sortie = bac.executer(parametres)
        resultat = sortie["resultat"]

        if not controler("le worker reussit", sortie["code"] == 0,
                         "code %s, journal : %s"
                         % (sortie["code"], sortie["journal"].strip()[-300:])):
            return True

        moteurs = dict((r, f) for r, f, _ in (resultat.get("moteurs") or []))
        # Compare au fichier reellement fourni, et non a NOM_GFPGAN : la
        # variable d'environnement sert justement a designer un fichier porte
        # ailleurs, sous un autre nom. Un controle sur le nom attendu virerait
        # au rouge pour cette seule raison, et masquerait ce qu'il est cense
        # etablir.
        controler("le modele d'amelioration est celui qui a servi",
                  moteurs.get("amelioration") == os.path.basename(gfpgan),
                  json.dumps(resultat.get("moteurs")))
        controler("aucune degradation n'est signalee",
                  not resultat.get("degradations"),
                  json.dumps(resultat.get("degradations")))
        controler("aucun echec de modele n'est signale",
                  not resultat.get("echecs_modeles"),
                  json.dumps(resultat.get("echecs_modeles")))

        visages = resultat.get("visages") or []
        if not controler("un visage est detecte et traite", len(visages) == 1,
                         json.dumps(visages)):
            return True

        apres = sortie["image"]
        if not controler("une image sort", apres is not None):
            return True
        controler("les dimensions sont conservees",
                  apres.shape[:2] == avant.shape[:2],
                  "%s -> %s" % (avant.shape, apres.shape))

        ecart = np.abs(apres[:, :, :3].astype(np.int32)
                       - avant[:, :, :3].astype(np.int32))
        masque = np.zeros(ecart.shape[:2], bool)
        for visage in visages:
            x, y, large, haut = [int(v) for v in visage["rect"][:4]]
            masque[max(0, y):y + haut, max(0, x):x + large] = True
        controler("le visage a bien ete modifie",
                  masque.any() and float(ecart[masque].mean()) > 0.5,
                  "ecart moyen %.2f" % (float(ecart[masque].mean())
                                        if masque.any() else 0.0))
        controler("rien n'a change hors de la zone du visage",
                  (~masque).any() and int(ecart[~masque].max()) == 0,
                  "ecart maximal %d" % (int(ecart[~masque].max())
                                        if (~masque).any() else -1))
    finally:
        bac.nettoyer()
    return True


def main():
    print("Modele d'amelioration - controles sur le vrai fichier")
    print("")
    execute = test_le_greffon_emploie_le_modele()
    print("")
    if not execute:
        print("Controles ignores : le modele n'est pas disponible.")
        return 0
    echecs = [nom for nom, ok, _ in RESULTATS if not ok]
    print("%d controles, %d echec(s)" % (len(RESULTATS), len(echecs)))
    return 1 if echecs else 0


if __name__ == "__main__":
    sys.exit(main())
