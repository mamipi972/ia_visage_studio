#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Lance tous les controles du depot et resume le resultat.

Usage : python3 outils/tous_les_tests.py

Code de retour non nul des qu'un controle echoue, pour servir tel quel dans
une procedure de livraison. Aucune livraison n'a lieu avec un controle en
echec : un controle rouge tolere une fois cesse d'etre un controle, et la fois
suivante personne ne saura distinguer l'echec benin de l'autre. Si le controle
se trompe, c'est le controle qu'on corrige, avant de livrer.
"""

import os
import subprocess
import sys

ICI = os.path.dirname(os.path.abspath(__file__))

ETAPES = [
    ("Controles de livraison", "verifier_livraison.py", []),
    ("Tests du greffon", "tests_unitaires.py", []),
    ("Tests du worker", "tests_worker.py", ["numpy", "cv2"]),
    ("Tests d'integration", "tests_integration.py", ["numpy", "cv2"]),
    ("Detecteur YuNet (vrai modele)", "tests_yunet.py", ["numpy", "cv2"]),
    ("Preuve par mutation", "tests_mutation.py", ["numpy", "cv2"]),
]


def dependances_presentes(modules):
    for nom in modules:
        try:
            __import__(nom)
        except ImportError:
            return False, nom
    return True, None


def main():
    resume = []
    echec = False
    for titre, script, dependances in ETAPES:
        disponibles, manquant = dependances_presentes(dependances)
        if not disponibles:
            resume.append((titre, "IGNORE", "module %s absent" % manquant))
            continue
        print("=" * 72)
        print(titre)
        print("=" * 72)
        code = subprocess.call([sys.executable, os.path.join(ICI, script)])
        print("")
        if code == 0:
            resume.append((titre, "OK", ""))
        else:
            resume.append((titre, "ECHEC", "code %d" % code))
            echec = True

    print("=" * 72)
    print("Resume")
    print("=" * 72)
    for titre, etat, detail in resume:
        print("  %-8s %s%s" % (etat, titre, (" (" + detail + ")") if detail else ""))
    if any(etat == "IGNORE" for _, etat, _ in resume):
        print("")
        print("  Les tests du worker, d'integration et la preuve par")
        print("  mutation demandent :")
        print("    pip install numpy opencv-python-headless")
    return 1 if echec else 0


if __name__ == "__main__":
    sys.exit(main())
