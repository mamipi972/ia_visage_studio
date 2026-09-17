#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Doublure generique de l'API GIMP 3.0, commune aux greffons de la suite.

Chacun appelle Gimp.main() des son chargement : sans cette doublure, aucun
d'eux ne peut etre importe pour etre teste. Elle ne simule rien d'autre que ce
qu'il faut pour que le module se charge et que Gimp.directory() reponde.
"""

import importlib.util
import os
import sys
import types

MESSAGES = []


class _Enum(object):
    def __init__(self, *noms):
        for nom in noms:
            setattr(self, nom, "<%s>" % nom)


class _Faux(object):
    """Objet permissif : tout attribut existe, tout appel repond."""

    def __init__(self, nom="faux"):
        self._nom = nom

    def __getattr__(self, item):
        if item.startswith("__"):
            raise AttributeError(item)
        return _Faux(self._nom + "." + item)

    def __call__(self, *args, **kwargs):
        return _Faux(self._nom + "()")


class FauxGimp(object):

    class PlugIn(object):
        __gtype__ = "FauxGType"

    class Selection(object):
        retour = None

        @classmethod
        def bounds(cls, image):
            if cls.retour is None:
                raise RuntimeError("bounds indisponible")
            return cls.retour

    RunMode = _Enum("INTERACTIVE", "NONINTERACTIVE", "WITH_LAST_VALS")
    PDBProcType = _Enum("PLUGIN")
    PDBStatusType = _Enum("SUCCESS", "CANCEL", "CALL_ERROR", "EXECUTION_ERROR")
    ProcedureSensitivityMask = _Enum("DRAWABLE")
    LayerMode = _Enum("NORMAL")
    AddMaskType = _Enum("WHITE", "ALPHA")
    ChannelOps = _Enum("REPLACE")
    FillType = _Enum("TRANSPARENT", "WHITE")
    ImageType = _Enum("RGBA_IMAGE", "RGB_IMAGE")

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
    def get_pdb():
        return _Faux("pdb")

    def __getattr__(self, item):
        return _Faux(item)


def _permissif(nom):
    """Classe dont tout attribut de classe manquant devient un faux objet."""

    class Meta(type):
        def __getattr__(cls, item):
            if item.startswith("__"):
                raise AttributeError(item)
            return _Faux(nom + "." + item)

    return Meta(nom, (object,), {})


class FauxGLib(object):
    class Error(Exception):
        pass

    @staticmethod
    def filename_to_uri(chemin):
        return "file://" + chemin.replace(os.sep, "/")

    @staticmethod
    def idle_add(*args, **kwargs):
        return 0


class FauxGObject(object):
    ParamFlags = _Enum("READWRITE")
    TYPE_STRING = "gchararray"
    TYPE_INT = "gint"
    TYPE_BOOLEAN = "gboolean"
    TYPE_DOUBLE = "gdouble"

    class GObject(object):
        pass


def installer(dossier_gimp, chemin_greffon, nom_module="greffon_sous_test"):
    """Injecte la doublure puis charge le greffon comme module."""
    FauxGimp.dossier = dossier_gimp
    MESSAGES[:] = []

    # Les classes de GIMP absentes de la doublure deviennent des faux objets,
    # pour qu'un greffon qui en utilise d'autres se charge quand meme.
    gimp = FauxGimp
    for nom in ("Image", "Layer", "GroupLayer", "Drawable", "Channel",
                "ImageProcedure", "Procedure", "Progress", "Display",
                "TextLayer", "Item", "Context"):
        if not hasattr(gimp, nom):
            setattr(gimp, nom, _permissif("Gimp." + nom))
    for nom in ("progress_init", "progress_set_text", "progress_pulse",
                "progress_update", "progress_end", "displays_flush",
                "context_push", "context_pop", "file_save", "file_load",
                "file_load_layer", "file_load_layers", "version",
                "get_version"):
        if not hasattr(gimp, nom):
            setattr(gimp, nom, staticmethod(_Faux("Gimp." + nom)))

    gi = types.ModuleType("gi")
    gi.require_version = lambda *a, **k: None
    repository = types.ModuleType("gi.repository")
    repository.Gimp = gimp
    repository.GimpUi = _permissif("GimpUi")
    repository.Gio = _permissif("Gio")
    repository.Gegl = _permissif("Gegl")
    repository.GObject = FauxGObject
    repository.GLib = FauxGLib
    gi.repository = repository
    sys.modules["gi"] = gi
    sys.modules["gi.repository"] = repository

    sys.modules.pop(nom_module, None)
    spec = importlib.util.spec_from_file_location(nom_module, chemin_greffon)
    module = importlib.util.module_from_spec(spec)
    sys.modules[nom_module] = module
    spec.loader.exec_module(module)
    return module
