# Outils de suite

Ces deux fichiers ne portent pas sur `ia_visage_studio` mais sur **les autres
greffons de la suite**, dont les correctifs ont été livrés sous forme de
fichiers `.py` et non par leurs dépôts. Ils sont conservés ici parce que
`AUDIT_DE_SUITE.md` vit dans ce dépôt, et qu'un contrôle qu'on ne retrouve pas
n'est pas rejoué.

| Fichier | Rôle |
| --- | --- |
| `doublure_gimp.py` | Doublure générique de l'API GIMP 3.0. Chaque greffon appelle `Gimp.main()` dès son chargement : sans elle, aucun ne peut être importé pour être testé. Plus permissive que `outils/faux_gimp.py`, parce qu'elle doit charger quatre greffons aux conventions différentes. |
| `tests_purge_suite.py` | Contrôles du correctif de purge des journaux sur `deep_erase`, `deep_erase_pro`, `gimp_sam2_segmentation` et `ia_detourage`, preuve par mutation comprise. |

## Usage

```
python3 outils/suite/tests_purge_suite.py <dossier contenant les quatre .py>
```

Les quatre greffons ne sont dans aucun dépôt accessible : le dossier qui les
contient se passe donc en argument. Sans argument, ils sont cherchés dans le
dossier parent du script.

Ces contrôles chargent réellement chaque greffon et exercent ses propres
fonctions — ce n'est pas une réimplémentation. Ils portent sur ce qui reste sur
le disque après la purge, jamais sur l'absence d'erreur.
