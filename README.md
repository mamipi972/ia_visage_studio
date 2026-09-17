# IA Visage Studio — greffon GIMP 3.0

Quatre traitements cumulatifs sur les visages d'une image, enchaînés en une
seule passe : **anonymiser**, **faire sourire**, **améliorer**, **coloriser**.

Le greffon installe lui-même son environnement Python, va chercher les modèles
manquants, et **dégrade au lieu de s'interrompre** quand un modèle n'est pas
disponible. **Aucune étape ne demande de taper une commande ni de supprimer un
dossier.**

> **Document bilingue.** La version française ci-dessous fait référence ; elle
> est complète et c'est elle qui est tenue à jour. Une version anglaise abrégée
> se trouve en fin de document : [**English version**](#english-version).
>
> **Bilingual document.** The French text below is the reference version: it is
> complete and kept up to date. A condensed English version is at the end of
> this file: [**English version**](#english-version).

---

## Installation

1. Dans le dossier des greffons de GIMP, **créer un dossier nommé
   `ia_visage_studio`** et y copier `ia_visage_studio.py` :

   | Système | Chemin |
   | --- | --- |
   | Windows | `%APPDATA%\GIMP\<version>\plug-ins\ia_visage_studio\` |
   | Linux | `~/.config/GIMP/<version>/plug-ins/ia_visage_studio/` |
   | macOS | `~/Library/Application Support/GIMP/<version>/plug-ins/ia_visage_studio/` |

   `<version>` est celle de votre GIMP : `3.0`, `3.2`...

   **Le dossier doit porter exactement le nom du fichier, sans le `.py`** :
   GIMP ne charge pas un greffon dont les deux noms diffèrent, et il ne le dit
   pas. Sur macOS et Linux, rendre ensuite le fichier exécutable (`chmod +x`).

   Dans ce dépôt, le fichier est à la racine : c'est à l'installation que le
   dossier se crée.

2. Redémarrer GIMP. Le filtre apparaît dans **Filtres > IA Suite > IA Visage
   Studio...**

Si le filtre n'apparaît pas, regarder si le fichier
`journal_ia_visage_studio.log` existe dans le dossier partagé (voir
[Où sont mes fichiers](#où-sont-mes-fichiers)). **Son absence est une
information** : elle prouve que GIMP n'a jamais exécuté le fichier, et oriente
vers le transport ou le nom du dossier plutôt que vers le code.

Le fichier livré est en ASCII pur et en fins de ligne LF. Publiez sa somme
SHA-256 à côté de lui (`python3 outils/verifier_livraison.py` la calcule) :
c'est le seul moyen de distinguer un greffon défectueux d'un fichier altéré
pendant le transport.

---

## Les quatre opérations

Chaque case correspond à une opération. Elles se cumulent, sauf celles que la
table de préséance exclut (voir plus bas).

| Case | Ce qu'elle fait | Modèle | Source | Sans le modèle |
| --- | --- | --- | --- | --- |
| **Anonymiser les visages** | Flou gaussien ou mosaïque sur chaque visage | aucun | — | fonctionne toujours |
| **Faire sourire les visages** | Modifie l'expression sur chaque vignette | `attgan_smile.onnx` | **à fournir** | l'étape est ignorée, et le motif affiché |
| **Améliorer les visages** | Restaure les détails de peau et d'yeux | `gfpgan_1_4.onnx` (~340 Mo) | déclarée, non vérifiée | rehaussement sans IA (masque flou + lissage) |
| **Coloriser l'image** | Colorise toute l'image, luminance conservée | `deoldify_artistic.onnx` | **à fournir** | l'étape est ignorée, et le motif affiché |

Toutes ces opérations ont d'abord besoin de **détecter les visages**, ce que le
greffon sait faire seul :

| Détecteur | Poids | Source | Quand |
| --- | --- | --- | --- |
| `face_detection_yunet_2023mar.onnx` | 227 Ko | **vérifiée** (zoo de modèles d'OpenCV, Apache-2.0) | par défaut ; téléchargé seul au premier usage |
| `yolov8n-face.onnx` | ~13 Mo | **à fournir** | s'il est déjà sur le disque — un fichier déposé l'a été délibérément, il passe donc devant |
| cascade de Haar | livrée avec OpenCV 4.x | — | si aucun modèle n'est disponible |
| votre sélection | — | — | si aucun détecteur n'aboutit |

**L'anonymisation ne dépend d'aucun modèle lourd et d'aucun réseau.**
C'est délibéré : c'est l'opération dont on a besoin quand on en a besoin, et
elle ne doit dépendre ni du réseau, ni d'un fichier de plusieurs centaines de
mégaoctets. Le détecteur par défaut pèse 227 Ko.

**« Source vérifiée » veut dire quelque chose de précis** : le fichier a été
téléchargé depuis cette adresse, chargé par OpenCV et exécuté depuis ce dépôt.
« Déclarée, non vérifiée » veut dire que l'adresse est écrite dans le code mais
que personne ne l'a jointe : le libellé de la case le dit, et l'opération
dégrade si elle ne répond pas. Deux adresses HuggingFace pour le détecteur
YOLOv8 figuraient ici et renvoyaient **HTTP 401** chez un utilisateur — dépôt
disparu ou devenu privé. Elles ont été retirées : une adresse dont on sait
qu'elle ne répond pas ne vaut pas mieux que pas d'adresse, et elle coûte une
requête à chaque lancement.

### Réglages

| Réglage | Valeurs | Effet |
| --- | --- | --- |
| Mode d'anonymisation | flou (défaut) / mosaïque | Les deux rayons sont proportionnels à la taille du visage : un rayon fixe laisserait un visage lisible sur une photo haute définition. |
| Intensité du sourire | 0,0 à 1,0 (défaut 1,0) | Valeur posée dans le vecteur d'attributs du modèle. |
| Nombre maximum de visages | 1 à 32 (défaut 32) | Au-delà, les visages les moins sûrs sont ignorés. |
| Télécharger les modèles manquants | coché par défaut | Téléchargement automatique sous 400 Mo par fichier, **taille annoncée avant de commencer**. |
| Utiliser la carte graphique | décoché par défaut | Installe un second environnement, nettement plus volumineux. Voir [Accélération matérielle](#accélération-matérielle). |
| Réinstaller l'environnement IA | décoché par défaut | Reconstruit l'environnement Python. C'est la sortie de secours si l'installation s'est mal passée — **il n'y a jamais de dossier à supprimer à la main**. |

### Si une sélection est active

Seuls les visages dont le centre tombe dans la sélection sont traités. Et si
aucun détecteur n'aboutit, la sélection elle-même est traitée comme un visage
unique : sur un poste sans réseau et sans cascade OpenCV, anonymiser une zone
choisie à la main reste possible.

---

## La table de préséance

Cocher **Anonymiser** en même temps que **Faire sourire** ou **Améliorer** n'a
pas de sens : il faudrait faire sourire un visage qu'on vient de flouter, et
l'ordre dans lequel les deux s'appliqueraient tiendrait du hasard.

**L'anonymisation est prioritaire.** Quand elle est cochée, le sourire et
l'amélioration sont retirés avant même que le sous-processus ne soit lancé, et
un message le dit :

> Sourire ignorée car l'anonymisation est prioritaire

La colorisation, qui porte sur l'image entière et sur une autre dimension,
n'entre en conflit avec rien : elle se cumule avec tout.

L'arbitrage a lieu dans le greffon. Le worker applique la même table une
seconde fois, et **signale** s'il a eu à le faire : si ce message apparaît,
c'est que l'arbitrage du greffon n'a pas fonctionné, et c'est un défaut à
signaler.

---

## L'ordre des étapes, et pourquoi

```
  colorisation (image entière)
        |
  détection des visages
        |
  anonymisation   OU   sourire puis amélioration
        |
  fusion des vignettes dans l'image
```

La colorisation passe **en premier** : les modèles de visage travaillent alors
sur une image déjà en couleur, comme ils l'attendent. L'inverse recolorierait
les visages qu'on vient de restaurer.

**Un seul modèle est chargé à la fois.** Chaque étape ouvre sa session, calcule
sur toutes les vignettes, puis relâche sa référence et attend que le pilote ait
réellement désalloué avant que la suivante ne charge quoi que ce soit.
L'empreinte mémoire à l'instant T ne dépasse donc jamais celle du modèle le
plus lourd de la chaîne — et non leur somme.

**Le format pivot.** Entre deux étapes, l'image revient toujours au même
format : tableau NumPy, **RGB**, **uint8**, 0-255. Chaque étape convertit ce
pivot vers ce dont son modèle a besoin, puis y revient avant de rendre la main.
Un oubli ici ne lève aucune exception : il produit un visage bleu. Le worker
vérifie donc le pivot en sortie de chaque étape et s'arrête **avant** de
produire un résultat faux, avec le marqueur `[ERR_PIVOT]`.

---

## Ce que le greffon garantit sur vos pixels

- **Hors des zones de visage, aucun pixel ne change.** Pas « presque aucun » :
  l'écart est exactement nul, y compris sur l'anneau extérieur de chaque zone,
  où le fondu atteint zéro. C'est vérifié en 8 et en 16 bits par
  `outils/tests_worker.py`.
- **La profondeur est restituée.** Une image 16 bits ressort en 16 bits. Une
  lecture naïve d'un PNG 16 bits produirait une image blanche sans lever la
  moindre exception ; c'est un des défauts que la preuve par mutation remet
  volontairement en place pour vérifier que le test sait le voir.
- **Les quatre cas de canaux sont gérés** — gris, gris + alpha, couleur,
  couleur + alpha — et la structure est restituée, à deux exceptions près,
  toutes deux annoncées : une image grise **colorisée** ressort en couleur (par
  définition), et une image **grise avec transparence** ressort en couleur avec
  les trois canaux identiques, parce qu'OpenCV n'écrit pas de PNG gris + alpha.
- **La colorisation ne déplace pas la luminance.** Seule la chrominance vient
  du modèle. C'est une propriété vérifiable, là où une sortie RGB reprise telle
  quelle peut éclaircir ou assombrir l'image sans que rien ne le signale.

Le calque produit est inséré **au-dessus** du calque traité, dans le même
groupe. Son nom dit ce qui a réellement servi, par exemple :

> `Portrait - detection yolov8n-face.onnx sur processeur, anonymisation OpenCV sur processeur`

Ce nom n'est pas décoratif. Le journal du worker disparaît avec le dossier
temporaire : c'est la seule trace persistante de ce qui a été employé, et les
valeurs viennent du worker, pas d'une déduction du greffon.

---

## Premier lancement

Tout se fait seul, avec la barre de progression animée :

1. **Recherche d'un Python système.** Registre Windows (deux ruches, deux
   arborescences, deux vues), lanceur `py`, emplacements d'installation
   standards, puis le `PATH` en dernier recours. Un Python livré avec une autre
   application (Blender, Inkscape...) est **déclassé mais pas rejeté** : sur un
   poste qui n'a rien d'autre, refuser reviendrait à ne pas fonctionner. Le
   canal retenu et la date sont notés dans le marqueur.
2. **Création d'un environnement virtuel dédié** et installation de `numpy`,
   `opencv-python-headless` et `onnxruntime` — les trois seules dépendances que
   le worker importe réellement.
3. **Téléchargement des modèles nécessaires aux cases cochées**, taille annoncée
   avant de commencer. Une case décochée ne télécharge rien. Seul le détecteur
   YuNet (227 Ko) a une adresse vérifiée ; les autres modèles se déposent à la
   main, et le message vous dit où.
4. **Traitement**, dans un processus séparé de GIMP.

Les lancements suivants sautent directement à l'étape 4 : un marqueur
d'environnement évite de revérifier les imports, ce qui coûterait deux à cinq
secondes avant tout travail utile.

### L'environnement est partagé avec le reste de la suite

Le greffon `gimp_sam2_segmentation` déclare exactement la même pile technique.
Les deux partagent donc `venv-onnx-cpu` : le premier qui s'installe fait le
travail, le second constate et démarre. Ce n'est pas une déduction de lecture
du code — c'est mesuré, et le contrôle est permanent : sur un venv déjà en
place et sans marqueur, le greffon lance zéro commande `pip` et zéro création
d'environnement (`outils/tests_unitaires.py`, `test_poste_deja_installe`).

**Le marqueur d'environnement, lui, est propre à chaque greffon**
(`env_onnx-cpu_ia_visage_studio.json`). C'est délibéré : il contient le numéro
de version du greffon, et un marqueur commun ferait qu'à chaque bascule entre
deux greffons de versions différentes, chacun jugerait l'environnement périmé
et le reconstruirait — plusieurs centaines de mégaoctets, indéfiniment, sans le
moindre message. `outils/verifier_livraison.py` relit les deux greffons
ensemble et refuse la livraison si leurs piles divergent.

---

## Où sont mes fichiers

Deux emplacements, et la distinction compte.

**Les données volumineuses** — environnement Python et modèles — sont hors du
profil GIMP, et leur chemin ne contient **aucun numéro de version de GIMP** :
une mise à jour de GIMP ne doit pas faire retélécharger plusieurs centaines de
mégaoctets ni les laisser immobilisés.

| Système | Chemin, collable tel quel |
| --- | --- |
| Windows | `%LOCALAPPDATA%\GIMP\ai_suite_shared\models` |
| Linux | `~/.local/share/GIMP/ai_suite_shared/models` |
| macOS | `~/Library/Application Support/GIMP/ai_suite_shared/models` |

**Ce n'est pas le dossier de profil de GIMP.** Sous Windows, le profil est dans
`AppData\Roaming` et les données ici dans `AppData\Local` : sur un poste
d'entreprise à profil itinérant, `Roaming` est synchronisé à chaque ouverture
et fermeture de session, et plusieurs gigaoctets y deviendraient un incident
d'exploitation.

**Les fichiers légers** — journal, marqueur, cache d'interpréteur, fichier de
confiance des modèles, inventaire, journaux d'incident — vivent sous
`<profil GIMP>/ai_suite_shared/`. Leur perte au changement de profil coûte une
revalidation de quelques secondes, une fois.

Le greffon écrit `inventaire_ia_visage_studio.json` dans le dossier partagé :
il liste ce qu'il a créé hors de son dossier temporaire, avec les tailles. Un
greffon qui fonctionne peut gaspiller en silence — sans inventaire, des
centaines de mégaoctets immobilisés ne se remarquent jamais.

Pour un autre disque ou une installation portable, la variable d'environnement
`GIMP_AI_SUITE_DIR` impose la racine des données. Elle n'est **jamais**
nécessaire.

---

## Où trouver les modèles

**Le détecteur, lui, s'obtient tout seul.** `face_detection_yunet_2023mar.onnx`
pèse 227 Ko, vient du zoo de modèles d'OpenCV, et son adresse a été vérifiée :
téléchargée, chargée et exécutée depuis ce dépôt le 2026-09-17. C'est le seul
modèle de la table dans ce cas, et c'est pourquoi c'est le seul que le greffon
va chercher de sa propre initiative.

Les trois autres adresses **n'ont pas pu être vérifiées** : le réseau du dépôt
de développement ne joint pas leurs hébergeurs. Le libellé de chaque case le
dit avant que vous ne la cochiez. Trois conséquences, toutes prévues :

- **Une adresse morte ne bloque rien.** L'opération dégrade — vers son chemin
  sans IA quand il en existe un, sinon elle est ignorée — et le motif est
  affiché en fin de traitement.
- **Le dépôt manuel fonctionne toujours.** Déposez le fichier attendu dans le
  dossier `models` ci-dessus ; le greffon le trouve, même rangé dans un
  sous-dossier, et le remonte à sa place canonique. Le chemin affiché dans le
  message est bien celui qu'il lit.
- **Une autre adresse s'ajoute sans toucher au code.** Créez
  `sources_modeles.json` dans le dossier partagé :

  ```json
  {
    "sourire": ["https://exemple.org/attgan_smile.onnx"],
    "colorisation": ["https://exemple.org/deoldify_artistic.onnx"]
  }
  ```

  Ces adresses passent devant celles du code. Ce fichier est facultatif : il
  existe pour qu'une adresse devenue morte se répare sans édition de code, ce
  que ce greffon ne demandera jamais.

Le message affiché en fin de traitement **donne ces deux chemins** — le dossier
de dépôt et celui de `sources_modeles.json` — pour chaque modèle manquant.
Constater une absence sans dire comment y remédier obligerait à quitter GIMP
pour chercher, et c'est précisément ce que ce greffon ne doit jamais demander.

**N'importe quel export ONNX convient** s'il respecte le format d'entrée du
tableau. Les quatre contrats sont dans
[`TABLE_DES_VALEURS.md`](TABLE_DES_VALEURS.md) : nom de fichier, taille
d'entrée, espace colorimétrique et normalisation. Ces valeurs ne se devinent
pas — une valeur raisonnable au jugé donne un résultat faux sans erreur.

### Intégrité des fichiers

Le greffon mémorise l'empreinte SHA-256 de chaque modèle à sa première
utilisation et **avertit** si elle change ensuite. Il n'y a **aucune empreinte
de référence et aucun blocage** : la provenance des poids varie selon
l'utilisateur, et un hash figé refuserait tout fichier légitime issu d'une
autre source. Ce contrôle détecte un changement, jamais une malveillance.

Avant le calcul de l'empreinte, un contrôle de taille et d'en-tête rejette ce
qui n'est manifestement pas un modèle : fichier vide ou tronqué, page HTML
d'erreur, JSON, archive, pointeur Git LFS. Un téléchargement interrompu porte
le bon nom et condamnerait l'opération à chaque lancement : il est supprimé et
retéléchargé — mais seulement dans un dossier que le greffon gère lui-même et
sous un nom de modèle connu.

---

## Accélération matérielle

La case **Utiliser la carte graphique** installe un second environnement
Python (`venv-onnx-gpu`, plusieurs gigaoctets). Elle est décochée par défaut, et
son libellé annonce ce coût.

**Elle est refusée avant tout téléchargement** si ce poste n'expose pas le
runtime CUDA. C'est le seul endroit où le greffon décline une demande explicite,
et il explique alors ce qui manque et comment s'en passer :

> onnxruntime-gpu n'embarque pas son runtime, contrairement aux roues de
> PyTorch : un pilote graphique ne suffit pas, il faut le CUDA Toolkit.

C'est aussi pourquoi la sonde porte sur le **runtime** et non sur le pilote :
sonder le pilote ferait basculer vers la roue GPU tous les postes qui n'ont
qu'un pilote — la grande majorité — pour un repli silencieux sur le processeur
après plusieurs gigaoctets téléchargés.

Quand le runtime est là, l'installation tente en plus les roues pip de cuDNN,
sans jamais en faire un échec bloquant, et expose leurs dossiers de
bibliothèques sur le `PATH` du worker — sans quoi elles seraient installées mais
introuvables.

**Le chemin est ensuite validé par une inférence réelle**, pas par une requête
de capacité : `onnxruntime` répond que `CUDAExecutionProvider` est disponible
même lorsque cuDNN manque, et l'échec ne surviendrait qu'au premier nœud de
convolution, au milieu de votre travail. Si l'inférence de contrôle échoue, le
traitement **se poursuit sur le processeur** et une phrase dit pourquoi. Un
message d'erreur à la place d'un calque serait un échec de conception : vous
vouliez traiter une image, pas arbitrer une question de pilotes.

Enfin, si le calcul s'est fait sur le processeur alors que la carte graphique
était demandée, le greffon le dit **une fois**, en joignant l'état des deux
couches — runtime du système et fournisseurs du moteur. Le message se déclenche
sur le symptôme observé, jamais sur l'hypothèse de sa cause : un avertissement
conditionné par « si un GPU a été détecté » ne s'afficherait pas quand la
détection échoue, c'est-à-dire précisément quand vous avez besoin d'être
informé.

---

## Quand ça se passe mal

**Les journaux sont archivés automatiquement.** En cas d'échec, journaux,
paramètres et script worker sont copiés dans un sous-dossier horodaté de
`logs/`, sous le dossier partagé, **avant** que le dossier de travail ne soit
détruit. Le message d'erreur cite ce chemin : joignez ce dossier à tout
signalement.

Ce dossier `logs/` est partagé par toute la suite, mais **le greffon ne purge
que ses propres archives** — celles qui portent un `incident.json` à son nom —
et il les date par le contenu, pas par le nom du dossier. Deux conventions de
nommage cohabitent dans la suite, et en ASCII le tiret précède le chiffre : une
purge alphabétique supprimerait systématiquement les archives des autres
greffons avant les siennes. Voir [`AUDIT_DE_SUITE.md`](AUDIT_DE_SUITE.md).

Une archive que personne ne revendique — antérieure à ce marqueur, ou produite
par un greffon qui ne le pose pas encore — n'est supprimée qu'à **deux
conditions réunies** : avoir plus de trente jours, et ne pas figurer parmi les
dix plus récentes d'entre elles. Le stock ancien se résorbe donc, sans qu'une
archive récente soit jamais menacée.

Vous n'avez **ni variable d'environnement à poser, ni terminal à ouvrir** pour
produire un rapport de bogue. Si le diagnostic en dépendait, il n'existerait
pas.

**Chaque échec porte son marqueur.** La liste complète, avec les codes de
sortie, est dans [`TABLE_DES_VALEURS.md`](TABLE_DES_VALEURS.md). Les plus
courants :

| Message | Ce qu'il faut faire |
| --- | --- |
| « aucun visage n'a été détecté » | Faire une sélection autour du visage : elle sert de repli au détecteur. |
| « l'environnement IA reste incomplet » | Cocher **Réinstaller l'environnement IA** et relancer. |
| « Espace disque insuffisant » | Le message chiffre le requis, le disponible et le chemin concerné. |
| « une étape a rendu l'image dans un format inattendu » | Défaut interne : le traitement s'est arrêté avant de produire un résultat faux. Joindre le dossier d'incident. |

Pour le développement, la variable `GIMP_AI_SUITE_DEBUG` conserve le dossier
d'exécution au lieu de le détruire. Elle ne remplace pas l'archivage
ci-dessus : elle le complète.

---

## Ce que le greffon ne garantit pas

- **La qualité des modèles n'est pas de son ressort.** Il applique l'export ONNX
  qu'on lui donne. Un modèle de sourire médiocre produira des sourires
  médiocres.
- **Une seule adresse de téléchargement a été vérifiée** : celle du détecteur
  YuNet. Les trois autres modèles sont à fournir, et le libellé de chaque case
  le dit avant que vous ne la cochiez. Voir
  [Où trouver les modèles](#où-trouver-les-modèles).
- **Le banc d'essai ne distingue pas une inversion RGB/BGR à l'entrée de
  YuNet.** Mesuré le 2026-09-17 : la permutation fait tomber le score de 0,90 à
  0,80 et décale la boîte de quelques pixels, sans jamais empêcher la
  détection. La conversion est faite selon le contrat d'OpenCV, mais aucun test
  ne saurait dire qu'elle a été oubliée.
- **Aucun repère de durée n'a été mesuré.** Les cases de
  [`TABLE_DES_VALEURS.md`](TABLE_DES_VALEURS.md) sont vides, et c'est voulu :
  une estimation plausible serait indiscernable d'une mesure.
- **Le gain réel de l'accélération matérielle n'a pas été mesuré.**
- **Aucun plafond mémoire n'est posé sous Windows.** Le plafond POSIX est posé
  et prouvé par un test qui pose une limite basse et vérifie le refus
  d'allocation. L'équivalent Windows — un Job Object — n'a pas pu être prouvé
  depuis ce dépôt, et une protection dont la correction n'est pas prouvée vaut
  moins que son absence.
- **La détection de l'annulation dépend de la build de GIMP.** L'API 3.0
  n'expose pas de façon stable, pour un greffon, l'état du bouton Annuler. Le
  greffon sonde plusieurs points d'entrée connus ; sur une build qui n'en
  expose aucun, l'annulation n'est pas détectée pendant le calcul. Ce qui est
  garanti dans tous les cas : **le processus enfant est tué quand le greffon
  rend la main**, quelle que soit la façon dont il sort — cela est testé.
- **Le repli d'API vers GIMP 4 n'est pas testé.** Une version majeure
  demandera une mise à jour du greffon. Vos données, elles, ne seront pas
  perdues : elles ne dépendent d'aucune version de GIMP.
- **Sur deux visages qui se chevauchent**, la zone commune reçoit le
  traitement du dernier visage composé.

### État de vérification

Les versions **1.0** de ce greffon **n'ont jamais été exécutées dans GIMP**.
Constat établi le **2026-09-16** par l'auteur du dépôt : l'environnement de
développement ne dispose ni de GIMP, ni des poids des modèles. Tout ce qui est
vérifié l'est hors de GIMP, par la doublure d'API et par les doublures de
moteur d'inférence — voir le tableau « Mesuré ou déclaré » de
[`TABLE_DES_VALEURS.md`](TABLE_DES_VALEURS.md).

Cette phrase porte une date et une version parce qu'elle se périmera : dès
qu'une exécution réelle aura eu lieu, elle doit être remplacée par ce qui a été
constaté, par qui, sur quel poste et avec quelle version.

---

## Développement et contrôles

```
python3 outils/tous_les_tests.py
```

Aucune livraison n'a lieu avec un contrôle en échec. Un contrôle rouge toléré
une fois cesse d'être un contrôle, et la fois suivante personne ne saura
distinguer l'échec bénin de l'autre. Si le contrôle se trompe, c'est le
contrôle qu'on corrige, avant de livrer.

| Outil | Ce qu'il couvre |
| --- | --- |
| `outils/verifier_livraison.py` | ASCII pur, fins de ligne, syntaxe du greffon et des scripts embarqués, cohérence des marqueurs et des clés de configuration entre le greffon et son worker, fidélité de la table des valeurs, cohérence avec le greffon voisin, somme SHA-256. |
| `outils/tests_unitaires.py` | Le greffon hors de GIMP, par la doublure d'API : arbitrage, découverte d'interpréteur sur une sortie réelle capturée, emplacements, marqueur, modèles face à un vrai serveur HTTP local, refus GPU, archivage, processus orphelins. |
| `outils/tests_worker.py` | Le worker de bout en bout, contre une doublure d'`onnxruntime` **qui vérifie le contrat des modèles et répond en fonction de son entrée**. |
| `outils/tests_integration.py` | `run_procedure` du début à la fin, hors de GIMP : vrai sous-processus, vrai fichier de paramètres, vrai calque produit, et l'enregistrement du greffon quand un appel d'API échoue. |
| `outils/tests_yunet.py` | Le détecteur par défaut, sur le **vrai** modèle : téléchargement, chargement par OpenCV, position des boîtes après remise à l'échelle. Ignoré, en le disant, si le modèle est injoignable. |
| `outils/tests_mutation.py` | La preuve par mutation : chaque correctif est remis en défaut, et son test doit alors échouer. |
| `outils/faux_gimp.py` | La doublure de l'API GIMP 3.0, une centaine de lignes. |
| `outils/sorties_reelles/` | Sorties réelles capturées, contre lesquelles les stratégies de détection sont testées. |
| [`AUDIT_DE_SUITE.md`](AUDIT_DE_SUITE.md) | La relecture croisée des cinq greffons de la suite : ressources partagées, écarts constatés, ce qui a été corrigé et où. À refaire avant toute nouvelle livraison dans la suite. |

### Pourquoi une preuve par mutation

Un test qui passe ne prouve rien tant qu'on n'a pas vérifié qu'il sait échouer.
`tests_mutation.py` remet trente-trois comportements fautifs dans une copie du
code et vérifie que le contrôle correspondant passe au rouge — oubli du
décalage de mise en lettre-boîte, pivot laissé en BGR, normalisation ignorée,
16 bits traité comme du 8 bits, table de préséance désactivée, plafond mémoire
retiré, calque inséré sous l'original, journaux non archivés...

Ce banc a payé immédiatement : à sa première exécution, **cinq contrôles qui
passaient ne prouvaient rien**, et deux de plus lors de l'ajout des mutations
d'intégration. L'un d'eux attribuait à la table de préséance un effet produit
en réalité par une structure `if/else` voisine — un troisième mécanisme
redondant qu'aucun test n'aurait pu distinguer des deux autres. Il a été retiré
du code pour que chaque mécanisme redevienne testable séparément. Deux autres
attribuaient à un mécanisme un message produit par un second : les contrôles
ont été réécrits pour porter sur la formulation que seul le mécanisme visé
produit.

Une mutation dont le motif n'existe plus est comptée **en échec**, jamais
ignorée en silence : une mutation périmée ne teste rien, et le dire est la
seule façon de ne pas s'en apercevoir trois versions plus tard.

### Ce que seul un poste réel peut couvrir

À reporter dans le tableau « Mesuré ou déclaré » de `TABLE_DES_VALEURS.md` :

- l'enregistrement du greffon dans GIMP, et son apparition dans le menu ;
- l'installation depuis un état vierge, sur un poste sans Python ou avec
  plusieurs versions de Python ;
- le comportement réel de chaque modèle, et la qualité des résultats ;
- les repères de durée, sur processeur et sur carte graphique ;
- la détection de l'annulation, qui dépend de la build de GIMP ;
- l'ordre d'empilement du calque produit sur un document à plusieurs calques et
  à groupes ;
- le comportement sur un profil GIMP itinérant sous Windows.

---

## English version

**IA Visage Studio** is a GIMP 3.0 plug-in that runs four cumulative face
operations in a single pass: **anonymise**, **add a smile**, **restore**, and
**colourise**.

**Install.** Create a folder named `ia_visage_studio` inside GIMP's `plug-ins`
directory and copy `ia_visage_studio.py` into it — the folder name must match
the file name exactly, or GIMP silently ignores it. Restart GIMP; the filter
appears under **Filters > IA Suite > IA Visage Studio...**

**First run.** The plug-in finds a system Python, builds its own virtual
environment (`numpy`, `opencv-python-headless`, `onnxruntime` — the only three
modules the worker imports), and downloads the models the ticked boxes need.
Download size is always announced before any transfer starts, and files above
400 MB are never fetched automatically. Nothing ever asks you to type a command
or delete a folder.

**Anonymisation needs no model and no download.** The other three degrade
gracefully: restoration falls back to a non-AI sharpening pass, while smile and
colourisation are skipped with a one-line explanation. Anonymisation takes
precedence over smile and restoration — ticking them together drops the latter
two before the subprocess is even launched.

**Your pixels.** Outside the face regions nothing changes — the difference is
exactly zero, verified in both 8-bit and 16-bit. Bit depth and channel layout
are restored, with two announced exceptions: a colourised greyscale image comes
back in colour, and greyscale-with-alpha comes back as colour with three
identical channels, because OpenCV cannot write greyscale+alpha PNG.
Colourisation changes chroma only; luminance is taken from the original.

**Where files live.** Large data (virtual environment and model weights) go to
`%LOCALAPPDATA%\GIMP\ai_suite_shared` on Windows, `~/.local/share/GIMP/…` on
Linux, `~/Library/Application Support/GIMP/…` on macOS — never in the roaming
GIMP profile, and never under a path containing a GIMP version number. Small
files (log, environment marker, model-trust file, inventory, incident logs)
live in the GIMP profile. `GIMP_AI_SUITE_DIR` overrides the data root; it is
never required.

**GPU.** The optional GPU stack lives in its own virtual environment. It is
refused before any download when the CUDA runtime is missing, and validated
afterwards by a real inference rather than a capability query — `onnxruntime`
reports CUDA as available even when cuDNN is absent. Any failure degrades to
CPU with a one-line reason instead of an error message.

**Not guaranteed.** Version 1.0 has never been run inside GIMP (stated
2026-09-16 by the repository author: the development environment has neither
GIMP nor the model weights). Only one download URL has been verified — the
default YuNet face detector (227 KB, from OpenCV's model zoo), downloaded,
loaded and run from this repository on 2026-09-17. The other three models must
be supplied by hand, and each checkbox says so before you tick it. No
timing figures have been measured, and the table in `TABLE_DES_VALEURS.md`
leaves those cells empty on purpose. There is no memory ceiling on Windows —
the POSIX one is in place and proven by a test. Cancellation detection depends
on the GIMP build; what is guaranteed is that the child process is killed when
the plug-in returns.

**Checks.** `python3 outils/tous_les_tests.py` runs delivery checks, plug-in
tests against a GIMP API double, worker tests against an `onnxruntime` double
that verifies each model's input contract, an end-to-end run of
`run_procedure` outside GIMP, and a mutation suite that puts twenty-eight
faults back into the code and requires the matching test to fail. Nothing
ships with a failing check.
