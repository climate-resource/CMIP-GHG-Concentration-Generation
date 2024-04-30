"""
Creation of configuration for processing NOAA's data
"""

from __future__ import annotations

base = dict()

GASES_SOURCES_DOWNLOAD_HASES = (
    (
        "co2",
        "surface-flask",
        "92d8a6a6c59d936f1b338c0bf781009cd25348bf9a2c8dd9dde3cbf21e8dfe17",
    ),
    (
        "co2",
        "in-situ",
        "0a68c9716bb9ec29e23966a2394e312618ed9b822885876d1ce5517bdf70acbe",
    ),
    (
        "ch4",
        "surface-flask",
        "e541578315328857f01eb7432b5949e39beabab2017c09e46727ac49ec728087",
    ),
    (
        "ch4",
        "in-situ",
        "c8ad74288d860c63b6a027df4d7bf6742e772fc4e3f99a4052607a382d7fefb2",
    ),
    (
        "n2o",
        "surface-flask",
        "6b7e09c37b7fa456ab170a4c7b825b3d4b9f6eafb0ff61a2a46554b0e63e84b1",
    ),
    (
        "sf6",
        "surface-flask",
        "376c78456bba6844cca78ecd812b896eb2f10cc6b8a9bf6cad7a52dc39e31e9a",
    ),
)
