# Fabriquer `gfpgan_1_4.onnx` soi-meme

## Pourquoi ce kit plutot que le fichier

Le fichier converti pese 325 Mo. Le canal par lequel je vous livre des
fichiers en refuse au-dela de 30 Mo : je ne peux pas vous l'envoyer. Ce
script le reconstruit chez vous, a l'identique quant a ce qui compte - les
memes poids officiels, la meme architecture, le meme controle numerique.

## Ce qu'il faut

Un Python 3.8 ou plus recent. Vous en avez un : c'est celui que le greffon a
deja trouve pour construire son environnement. Sinon, python.org, en cochant
*Add python.exe to PATH*.

Il faut aussi, le temps de l'operation, 5 Go de disque libre et 4 Go de
memoire vive. Comptez un quart d'heure.

## Comment

Posez `convertir_gfpgan.py` ou vous voulez, puis :

    py convertir_gfpgan.py          (Windows)
    python3 convertir_gfpgan.py     (Linux, macOS)

Il n'y a rien d'autre a taper. Le script cree son environnement, y installe
torch et onnx, telecharge les poids officiels GFPGAN v1.4 depuis les Releases
GitHub du projet, exporte, verifie, et depose le fichier a l'endroit meme ou
le greffon ira le chercher - y compris si votre installation utilise un
dossier de donnees different de l'emplacement canonique : il lit pour cela le
marqueur ecrit par le greffon.

Ensuite : relancez GIMP, cochez **Ameliorer les visages**. Le message
d'indisponibilite ne doit plus paraitre.

## Ce qui est verifie, et ce qui ne l'est pas

Verifie, et le script s'arrete si l'un des trois echoue :

1. les poids chargent sans une cle manquante ni une cle en trop - donc
   l'architecture et la configuration sont les bonnes, et non pas seulement
   compatibles ;
2. le fichier exporte, rejoue sous onnxruntime sur la meme entree, rend la
   meme chose que la reference PyTorch (ecart accepte : 0,001 ; mesure ici :
   0,000008) ;
3. la sortie a la forme `[1, 3, 512, 512]` qu'attend le greffon.

Non verifie, et c'est a savoir : que le resultat *vous plaise*. GFPGAN
reconstruit un visage ; sur une photo deja nette il peut lisser plus que vous
ne le souhaitez. C'est le modele, pas la conversion.

## Le SHA-256 du resultat n'est pas une reference

Deux machines ne produisent pas le meme fichier a l'octet pres : la version
de torch change le nom des noeuds et l'ordre des initialiseurs. Sur deux
exports ici, meme poids d'origine et meme ecart numerique, les tailles
differaient de 3 ko. Comparer votre empreinte a la mienne ne prouverait donc
rien - c'est le controle numerique du point 2 qui atteste du fichier, et il
tourne chez vous.

## Apres coup

Le dossier `venv-conversion` et le dossier `travail_conversion` sont
supprimables. Ils ne servaient qu'a la conversion ; le greffon n'en a pas
besoin. C'est tout l'interet d'avoir converti une fois pour toutes.

## Les deux autres modeles

`attgan_smile.onnx` (sourire) et `deoldify_artistic.onnx` (colorisation)
n'ont pas d'equivalent aussi direct : leurs poids publics ne se laissent pas
exporter par le meme chemin court. Les cases correspondantes restent donc
inoperantes, et le greffon le dit plutot que de le laisser deviner.
