#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Doublure minimale de l'API GIMP 3.0, pour tester le greffon hors de GIMP.

Le greffon appelle Gimp.main() des son chargement : sans cette doublure, il
est impossible de l'importer pour tester quoi que ce soit. Une centaine de
lignes suffisent a exercer la decouverte d'interpreteur, la resolution des
modeles, les marqueurs, l'arbitrage des intentions, l'archivage des journaux
et les messages - sans quoi tout se jouerait sur le poste de l'utilisateur.
"""

import importlib.util
import os
import sys
import types

MESSAGES = []


class _Enum:
    def __init__(self, *noms):
        for nom in noms:
            setattr(self, nom, "<%s>" % nom)


class _FauxObjet:
    def __init__(self, nom="faux"):
        self._nom = nom

    def __getattr__(self, item):
        if item.startswith("__"):
            raise AttributeError(item)
        return _FauxObjet(self._nom + "." + item)

    def __call__(self, *args, **kwargs):
        return _FauxObjet(self._nom + "()")


class FauxGimp:
    """Attributs utilises par le greffon. Les tests remplacent ce dont ils ont
    besoin (Selection.bounds, par exemple)."""

    class PlugIn(object):
        # GObject expose __gtype__ sur toute classe enregistree ; le greffon
        # le passe a Gimp.main() des son chargement.
        __gtype__ = "FauxGType"

    class Selection(object):
        retour = None

        @classmethod
        def bounds(cls, image):
            if cls.retour is None:
                raise RuntimeError("bounds indisponible")
            return cls.retour

    RunMode = _Enum("INTERACTIVE", "NONINTERACTIVE")
    PDBProcType = _Enum("PLUGIN")
    PDBStatusType = _Enum("SUCCESS", "CANCEL", "CALL_ERROR", "EXECUTION_ERROR")
    ProcedureSensitivityMask = _Enum("DRAWABLE")
    LayerMode = _Enum("NORMAL")
    AddMaskType = _Enum("WHITE")

    dossier = None

    @staticmethod
    def main(*args, **kwargs):
        return None

    @classmethod
    def directory(cls):
        return cls.dossier

    @staticmethod
    def message(texte):
        MESSAGES.append(texte)

    @staticmethod
    def progress_init(texte=None):
        return None

    @staticmethod
    def progress_set_text(texte=None):
        return None

    @staticmethod
    def progress_pulse():
        return None

    @staticmethod
    def progress_update(fraction):
        return None

    @staticmethod
    def progress_end():
        return None

    @staticmethod
    def displays_flush():
        return None

    @staticmethod
    def get_pdb():
        return _FauxObjet("pdb")

    @staticmethod
    def file_save(*args, **kwargs):
        raise RuntimeError("file_save indisponible dans la doublure")

    @staticmethod
    def file_load_layer(*args, **kwargs):
        raise RuntimeError("file_load_layer indisponible dans la doublure")

    @staticmethod
    def file_load_layers(*args, **kwargs):
        raise RuntimeError("file_load_layers indisponible dans la doublure")

    @staticmethod
    def file_load(*args, **kwargs):
        raise RuntimeError("file_load indisponible dans la doublure")

    ImageProcedure = _FauxObjet("ImageProcedure")
    Image = _FauxObjet("Image")
    Layer = _FauxObjet("Layer")
    GroupLayer = _FauxObjet("GroupLayer")


def installer_sonde_annulation(retour):
    """Expose Gimp.progress_get_cancelled, que certaines builds n'ont pas.

    Le greffon sonde plusieurs noms et se passe de tous : ce hook permet de
    tester les deux situations, celle ou la sonde existe et celle ou elle
    n'existe pas, sans supposer laquelle est la vraie.
    """
    setattr(FauxGimp, "progress_get_cancelled", staticmethod(lambda: retour))


def retirer_sonde_annulation():
    for nom in ("progress_get_cancelled", "progress_is_cancelled",
                "progress_cancelled"):
        if hasattr(FauxGimp, nom):
            delattr(FauxGimp, nom)


class FauxGLib:
    @staticmethod
    def filename_to_uri(chemin):
        return "file://" + chemin.replace(os.sep, "/")

    class Error(Exception):
        pass


class FauxGObject:
    ParamFlags = _Enum("READWRITE")


def installer(dossier_gimp, chemin_greffon):
    """Injecte la doublure puis charge le greffon comme module."""
    FauxGimp.dossier = dossier_gimp
    MESSAGES[:] = []

    gi = types.ModuleType("gi")
    gi.require_version = lambda *a, **k: None
    repository = types.ModuleType("gi.repository")
    repository.Gimp = FauxGimp
    repository.GimpUi = _FauxObjet("GimpUi")
    repository.Gio = _FauxObjet("Gio")
    repository.GObject = FauxGObject
    repository.GLib = FauxGLib
    gi.repository = repository
    sys.modules["gi"] = gi
    sys.modules["gi.repository"] = repository

    nom = "greffon_sous_test"
    sys.modules.pop(nom, None)
    spec = importlib.util.spec_from_file_location(nom, chemin_greffon)
    module = importlib.util.module_from_spec(spec)
    sys.modules[nom] = module
    spec.loader.exec_module(module)
    return module
