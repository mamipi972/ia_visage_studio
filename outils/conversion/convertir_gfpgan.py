#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Fabrique gfpgan_1_4.onnx pour le greffon IA Visage Studio.

Usage : python3 convertir_gfpgan.py
        (ou, sous Windows : py convertir_gfpgan.py)

Aucune adresse publique ne sert cet export ONNX : les trois essayees
renvoyaient 401 ou 404. Les poids officiels PyTorch, eux, sont publies sur les
Releases GitHub de GFPGAN et repondent. Ce script fait le pont, une seule fois.

Il ne demande rien. Il cree son propre environnement Python a cote de lui,
y installe ce qu'il faut, telecharge les poids, exporte, VERIFIE le resultat,
puis depose le fichier la ou le greffon le cherche. Vous n'avez pas une
commande a taper ni un dossier a creer.

Cet environnement est jetable. Il pese environ 300 Mo sous Windows et macOS,
ou la roue PyTorch publiee sur PyPI est une roue processeur. Sous Linux, la
meme roue embarque les bibliotheques CUDA et pese plusieurs gigaoctets : le
script y demande d'abord l'index processeur officiel de PyTorch, et ne
retombe sur PyPI que s'il ne repond pas. Dans les deux cas, le dossier
venv-conversion est supprimable une fois le modele produit : le greffon n'en
a pas besoin - c'est tout l'interet de convertir une fois pour toutes.

Comptez un quart d'heure, 5 Go libres pendant l'operation et 4 Go de memoire
vive : l'export tient le graphe entier en memoire avant de l'ecrire.

Ce que le script verifie, et qui fait la difference entre un fichier ONNX et
un fichier ONNX correct :

  - les poids chargent sans cle manquante ni cle en trop, donc l'architecture
    et la configuration sont les bonnes ;
  - le modele exporte, rejoue sous onnxruntime sur la meme entree, rend la
    meme chose que la reference PyTorch ;
  - la sortie a la forme et l'intervalle qu'attend le greffon.

Contrat d'entree et de sortie, lu dans le code d'inference de GFPGAN et non
supppose : RGB, 512x512, NCHW, normalise par (x/255 - 0,5) / 0,5, soit
l'intervalle [-1, 1]. La sortie est dans le meme intervalle et le meme espace.
"""

import glob
import hashlib
import json
import os
import subprocess
import sys
import time

VERSION = "1.0"

DEPOT_ARCH = "https://raw.githubusercontent.com/TencentARC/GFPGAN/master/gfpgan/archs/"
FICHIERS_ARCH = ("gfpganv1_clean_arch.py", "stylegan2_clean_arch.py")
URL_POIDS = ("https://github.com/TencentARC/GFPGAN/releases/download/"
             "v1.3.0/GFPGANv1.4.pth")
NOM_POIDS = "GFPGANv1.4.pth"
NOM_SORTIE = "gfpgan_1_4.onnx"
TAILLE = 512
OPSETS = (17, 16, 14, 13, 11)
ECART_MAX_ACCEPTE = 1e-3

PAQUETS = ["onnx", "onnxruntime>=1.16,<2", "numpy>=1.24,<3"]
INDEX_TORCH_CPU = "https://download.pytorch.org/whl/cpu"
NOM_VENV = "venv-conversion"
MARQUEUR_ENFANT = "IA_VISAGE_CONVERSION_ENFANT"

# Le greffon lit ces trois noms de la meme facon : ils decident de l'endroit
# ou le modele doit atterrir pour qu'il le trouve.
SHARED_DIR_NAME = "ai_suite_shared"
MOTIF_MARQUEUR = "env_*_ia_visage_studio.json"
VARIABLE_DOSSIER = "GIMP_AI_SUITE_DIR"

ICI = os.path.dirname(os.path.abspath(__file__))


def dit(texte=""):
    print(texte)
    sys.stdout.flush()


def empreinte(chemin):
    """SHA-256 lu par blocs.

    Ces fichiers pesent des centaines de megaoctets : les charger d'un bloc en
    memoire pour les hacher ferait echouer la conversion sur une machine juste
    en memoire, et pour rien.
    """
    calcul = hashlib.sha256()
    with open(chemin, "rb") as flux:
        while True:
            bloc = flux.read(1024 * 1024)
            if not bloc:
                break
            calcul.update(bloc)
    return calcul.hexdigest()


# ---------------------------------------------------------------------------
# Environnement dedie
# ---------------------------------------------------------------------------
def python_du_venv(dossier):
    if os.name == "nt":
        return os.path.join(dossier, "Scripts", "python.exe")
    return os.path.join(dossier, "bin", "python")


def installer_pile(py):
    """Installe torch, puis le reste.

    torch est demande a part parce que sa roue PyPI n'a pas le meme poids
    partout : roue processeur sous Windows et macOS, roue chargee des
    bibliotheques CUDA sous Linux. Faire venir plusieurs gigaoctets de CUDA
    pour un export qui se fait sur le processeur serait du gaspillage pur :
    sous Linux on demande donc d'abord l'index processeur officiel. S'il ne
    repond pas - reseau filtre, miroir d'entreprise - PyPI reste la reponse
    correcte, seulement plus lourde. Une conversion qui aboutit vaut mieux
    qu'une conversion econome qui echoue.
    """
    base = [py, "-m", "pip", "install", "--upgrade", "--only-binary=:all:"]
    dit("Installation de la pile de conversion (une seule fois).")
    code = 1
    if os.name != "nt" and sys.platform != "darwin":
        dit("  torch, depuis l'index processeur de PyTorch...")
        # Deux essais courts, pas cinq longs : cet index est une economie, pas
        # une dependance. Derriere un reseau qui le filtre, ses reprises par
        # defaut font attendre une minute pour un repli decide d'avance.
        code = subprocess.call(base + ["--retries", "2", "--timeout", "15",
                                       "--index-url", INDEX_TORCH_CPU, "torch"])
        if code != 0:
            dit("  index processeur injoignable ; reprise sur PyPI.")
    if code != 0:
        dit("  torch, depuis PyPI...")
        code = subprocess.call(base + ["torch"])
    if code != 0:
        raise SystemExit("L'installation de torch a echoue (code %s)." % code)
    dit("  " + ", ".join(PAQUETS) + "...")
    code = subprocess.call(base + PAQUETS)
    if code != 0:
        raise SystemExit("L'installation a echoue (code %s)." % code)


def assurer_environnement():
    """Cree le venv, y installe la pile, et s'y relance. Retourne True si le
    travail doit se poursuivre dans ce processus."""
    if os.environ.get(MARQUEUR_ENFANT):
        return True
    try:
        import torch, onnx, onnxruntime, numpy       # noqa: F401
        dit("Pile deja disponible dans cet interpreteur.")
        return True
    except ImportError:
        pass

    dossier = os.path.join(ICI, NOM_VENV)
    py = python_du_venv(dossier)
    if not os.path.isfile(py):
        dit("Creation d'un environnement dedie dans %s ..." % dossier)
        code = subprocess.call([sys.executable, "-m", "venv", dossier])
        if code != 0 or not os.path.isfile(py):
            raise SystemExit(
                "La creation de l'environnement a echoue (code %s).\n"
                "Verifiez que le module venv est disponible pour %s."
                % (code, sys.executable))

    installer_pile(py)

    dit("Reprise dans l'environnement dedie.")
    dit("")
    env = dict(os.environ)
    env[MARQUEUR_ENFANT] = "1"
    raise SystemExit(subprocess.call([py, os.path.abspath(__file__)], env=env))


# ---------------------------------------------------------------------------
# Approvisionnement
# ---------------------------------------------------------------------------
def telecharger(url, cible, libelle):
    import urllib.request

    if os.path.isfile(cible) and os.path.getsize(cible) > 0:
        dit("%s deja present (%s octets)." % (libelle, os.path.getsize(cible)))
        return cible
    dit("Telechargement de %s ..." % libelle)
    requete = urllib.request.Request(url)
    requete.add_header("User-Agent", "ia-visage-conversion/" + VERSION)
    recus = 0
    debut = time.time()
    partiel = cible + ".part"
    total = 0
    try:
        with urllib.request.urlopen(requete, timeout=120) as reponse:
            total = int(reponse.headers.get("Content-Length") or 0)
            with open(partiel, "wb") as flux:
                while True:
                    bloc = reponse.read(1024 * 512)
                    if not bloc:
                        break
                    flux.write(bloc)
                    recus += len(bloc)
                    if total:
                        sys.stdout.write("\r  %5.1f %%  (%d Mo)"
                                         % (100.0 * recus / total, recus // 1048576))
                        sys.stdout.flush()
    except BaseException:
        # Un fichier a moitie telecharge qui subsiste serait repris tel quel au
        # prochain lancement, et l'echec deviendrait incomprehensible.
        if total:
            dit("")
        try:
            os.remove(partiel)
        except OSError:
            pass
        raise
    if total:
        dit("")
    if total and recus != total:
        os.remove(partiel)
        raise SystemExit("%s est arrive tronque (%d octets sur %d attendus)."
                         % (libelle, recus, total))
    os.replace(partiel, cible)
    dit("  %d octets en %.0f s" % (recus, time.time() - debut))
    return cible


BOUCHON_ARCH_UTIL = '''"""Bouchon de basicsr.archs.arch_util.

L'architecture de GFPGAN n'en importe que default_init_weights, appelee a la
construction. Comme un etat entraine est charge par-dessus, l'initialisation
n'a aucune influence sur le resultat. L'installation de basicsr est cassee sur
PyPI : l'eviter n'est pas un contournement, c'est la seule voie.
"""


def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    return module_list
'''

BOUCHON_REGISTRY = '''"""Bouchon de basicsr.utils.registry : un decorateur qui rend la classe."""


class _Registre(object):
    def __init__(self, nom):
        self.nom = nom
        self.contenu = {}

    def register(self, obj=None):
        if obj is None:
            return lambda cible: self.register(cible)
        self.contenu[obj.__name__] = obj
        return obj

    def get(self, nom):
        return self.contenu[nom]


ARCH_REGISTRY = _Registre("arch")
'''


def preparer_sources(travail):
    """Depose l'architecture et les bouchons, et rend le dossier a ajouter au
    chemin d'import."""
    racine = os.path.join(travail, "sources")
    for sous in ("basicsr/archs", "basicsr/utils", "gfpgan_archs"):
        os.makedirs(os.path.join(racine, sous), exist_ok=True)
    for chemin in ("basicsr/__init__.py", "basicsr/archs/__init__.py",
                   "basicsr/utils/__init__.py", "gfpgan_archs/__init__.py"):
        open(os.path.join(racine, chemin), "w").close()
    with open(os.path.join(racine, "basicsr/archs/arch_util.py"), "w",
              encoding="utf-8") as f:
        f.write(BOUCHON_ARCH_UTIL)
    with open(os.path.join(racine, "basicsr/utils/registry.py"), "w",
              encoding="utf-8") as f:
        f.write(BOUCHON_REGISTRY)
    for nom in FICHIERS_ARCH:
        telecharger(DEPOT_ARCH + nom,
                    os.path.join(racine, "gfpgan_archs", nom),
                    "l'architecture %s" % nom)
    return racine


# ---------------------------------------------------------------------------
# Ou le greffon cherche ses modeles
# ---------------------------------------------------------------------------
def base_donnees():
    """Racine des donnees volumineuses, suivant les conventions du systeme.

    Copie conforme de la fonction du greffon : c'est lui qui decide ou il
    cherche, ce script ne fait que s'y ranger.
    """
    if os.name == "nt":
        return os.environ.get("LOCALAPPDATA",
                              os.path.expanduser("~\\AppData\\Local"))
    if sys.platform == "darwin":
        return os.path.expanduser("~/Library/Application Support")
    return os.environ.get("XDG_DATA_HOME", os.path.expanduser("~/.local/share"))


def profils_gimp():
    """Dossiers de profil GIMP, ou vivent les marqueurs d'installation."""
    if os.name == "nt":
        base = os.environ.get("APPDATA",
                              os.path.expanduser("~\\AppData\\Roaming"))
    elif sys.platform == "darwin":
        base = os.path.expanduser("~/Library/Application Support")
    else:
        base = os.environ.get("XDG_CONFIG_HOME", os.path.expanduser("~/.config"))
    return sorted(glob.glob(os.path.join(base, "GIMP", "*")))


def dossier_memorise():
    """Dossier de donnees note par une installation precedente du greffon.

    Le marqueur vit avec le profil GIMP et contient le chemin reellement
    retenu. Le lire est ce qui evite l'erreur la plus bete possible : deposer
    un fichier de 325 Mo a l'emplacement canonique alors que le greffon, lui,
    en utilise un autre - profil deplace, lettre de lecteur differente. Le
    modele serait la, correct, et invisible.
    """
    for profil in profils_gimp():
        for chemin in sorted(glob.glob(os.path.join(profil, SHARED_DIR_NAME,
                                                    MOTIF_MARQUEUR))):
            try:
                with open(chemin, "r", encoding="utf-8", errors="replace") as f:
                    donnees = json.load(f)
            except Exception:
                continue
            if not isinstance(donnees, dict):
                continue
            retenu = donnees.get("dossier_donnees")
            if retenu and os.path.isdir(retenu):
                return retenu
    return None


def dossier_modeles():
    """Ou le greffon ira chercher le modele, et pourquoi la.

    L'ordre reproduit celui du greffon, et doit le reproduire : la variable
    d'environnement, puis l'emplacement canonique s'il existe deja, puis
    celui qu'une installation precedente a memorise, puis un dossier laisse
    par une version anterieure. Rend le chemin et la raison, parce qu'un
    chemin inattendu affiche sans son motif ne s'explique pas.
    """
    force = os.environ.get(VARIABLE_DOSSIER, "").strip()
    if force:
        return os.path.join(force, "models"), "variable " + VARIABLE_DOSSIER

    base = base_donnees()
    canonique = os.path.join(base, "GIMP", SHARED_DIR_NAME)
    if os.path.isdir(canonique):
        return os.path.join(canonique, "models"), "emplacement canonique"

    memorise = dossier_memorise()
    if memorise:
        return (os.path.join(memorise, "models"),
                "dossier memorise par une installation precedente")

    for entree in sorted(glob.glob(os.path.join(base, "GIMP", "*",
                                                SHARED_DIR_NAME))):
        if os.path.isdir(entree):
            return os.path.join(entree, "models"), "dossier d'une version anterieure"

    return os.path.join(canonique, "models"), "emplacement canonique, a creer"


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------
def convertir(travail):
    racine = preparer_sources(travail)
    sys.path.insert(0, racine)

    import numpy as np
    import torch
    from gfpgan_archs.gfpganv1_clean_arch import GFPGANv1Clean

    poids = telecharger(URL_POIDS, os.path.join(travail, NOM_POIDS),
                        "les poids officiels GFPGAN v1.4 (332 Mo)")
    dit("  SHA-256 des poids : " + empreinte(poids))

    dit("")
    dit("Chargement du modele...")
    etat = torch.load(poids, map_location="cpu", weights_only=True)
    cles = etat.get("params_ema", etat.get("params", etat))
    modele = GFPGANv1Clean(
        out_size=TAILLE, num_style_feat=512, channel_multiplier=2,
        decoder_load_path=None, fix_decoder=False, num_mlp=8,
        input_is_latent=True, different_w=True, narrow=1, sft_half=True)
    manquants, en_trop = modele.load_state_dict(cles, strict=False)
    manquants = [c for c in manquants if "stylegan_decoder.noises" not in c]
    dit("  cles manquantes : %d, cles en trop : %d" % (len(manquants), len(en_trop)))
    if manquants or en_trop:
        dit("  ATTENTION : l'architecture ne correspond pas exactement aux "
            "poids. Le resultat serait faux ; arret.")
        return 1
    modele.eval()

    class Enveloppe(torch.nn.Module):
        """Une entree, une sortie, et un bruit fige.

        La forward d'origine rend un couple et accepte des drapeaux. Surtout,
        son bruit aleatoire rendrait le modele non deterministe : l'ecart avec
        la reference ne voudrait alors rien dire.
        """

        def __init__(self, interne):
            super(Enveloppe, self).__init__()
            self.interne = interne

        def forward(self, x):
            sortie = self.interne(x, return_rgb=False, randomize_noise=False)
            return sortie[0] if isinstance(sortie, (tuple, list)) else sortie

    enveloppe = Enveloppe(modele).eval()
    generateur = torch.Generator().manual_seed(1234)
    exemple = (torch.rand((1, 3, TAILLE, TAILLE), generator=generateur) * 2.0) - 1.0

    cible = os.path.join(travail, NOM_SORTIE)
    dit("")
    retenu = None
    for opset in OPSETS:
        try:
            dit("Export ONNX, opset %d..." % opset)
            torch.onnx.export(enveloppe, exemple, cible, opset_version=opset,
                              input_names=["input"], output_names=["output"],
                              do_constant_folding=True, dynamo=False)
            retenu = opset
            break
        except TypeError:
            try:
                torch.onnx.export(enveloppe, exemple, cible,
                                  opset_version=opset, input_names=["input"],
                                  output_names=["output"],
                                  do_constant_folding=True)
                retenu = opset
                break
            except Exception as e:
                dit("  echec : %s" % str(e)[:200])
        except Exception as e:
            dit("  echec : %s" % str(e)[:200])
    if retenu is None:
        dit("Aucun opset n'a permis l'export.")
        return 1
    dit("  export reussi en opset %d" % retenu)

    dit("")
    dit("Verification du fichier produit...")
    import onnxruntime as ort

    with torch.no_grad():
        reference = enveloppe(exemple).cpu().numpy()
    options = ort.SessionOptions()
    options.enable_mem_pattern = False
    options.enable_cpu_mem_arena = False
    session = ort.InferenceSession(cible, sess_options=options,
                                   providers=["CPUExecutionProvider"])
    entree = session.get_inputs()[0]
    sortie = session.get_outputs()[0]
    obtenu = session.run(None, {entree.name: exemple.cpu().numpy()})[0]
    ecart = float(np.abs(reference - obtenu).max())

    dit("  entree  : %s %s" % (entree.name, entree.shape))
    dit("  sortie  : %s %s" % (sortie.name, sortie.shape))
    dit("  ecart maximal avec la reference PyTorch : %.8f" % ecart)
    dit("  intervalle de sortie : [%.3f, %.3f]"
        % (float(obtenu.min()), float(obtenu.max())))
    if list(sortie.shape) != [1, 3, TAILLE, TAILLE]:
        dit("  ATTENTION : la forme de sortie n'est pas celle qu'attend le "
            "greffon. Fichier non installe.")
        return 1
    if ecart > ECART_MAX_ACCEPTE:
        dit("  ATTENTION : l'ecart avec la reference est trop grand. "
            "Fichier non installe.")
        return 1
    dit("  Le fichier exporte reproduit la reference PyTorch.")

    destination_dossier, raison = dossier_modeles()
    dit("")
    dit("Installation dans %s (%s)." % (destination_dossier, raison))
    os.makedirs(destination_dossier, exist_ok=True)
    destination = os.path.join(destination_dossier, NOM_SORTIE)
    import shutil
    shutil.move(cible, destination)
    empreinte_onnx = empreinte(destination)

    dit("")
    dit("=" * 70)
    dit("Termine.")
    dit("  Fichier  : " + destination)
    dit("  Taille   : %d octets" % os.path.getsize(destination))
    dit("  SHA-256  : " + empreinte_onnx)
    dit("")
    dit("Relancez GIMP et cochez \"Ameliorer les visages\".")
    dit("Le dossier %s peut etre supprime : le greffon n'en a pas besoin."
        % NOM_VENV)
    dit("=" * 70)
    return 0


def main():
    dit("Conversion GFPGAN v1.4 vers ONNX, pour IA Visage Studio (v%s)" % VERSION)
    dit("")
    assurer_environnement()
    travail = os.path.join(ICI, "travail_conversion")
    os.makedirs(travail, exist_ok=True)
    try:
        return convertir(travail)
    finally:
        dit("")
        dit("Fichiers de travail conserves dans %s (supprimables)." % travail)


if __name__ == "__main__":
    sys.exit(main())
