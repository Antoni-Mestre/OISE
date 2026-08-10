from __future__ import annotations

import hashlib
import html
import json
import math
import platform
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import FeatureUnion
from sklearn.model_selection import StratifiedShuffleSplit

from config import (
    CONFIGS,
    COARSE_TYPES,
    EXECUTION_DATE,
    FACT_LABELS,
    GLOBAL_SEED,
    PREDICTIONS,
    PROCESSED,
    RAW,
    RESOURCE_LABELS,
    ROOT,
    STATISTICS,
)


TAG_RE = re.compile(r"<[^>]+>")
SPACE_RE = re.compile(r"\s+")


def clean_html(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = html.unescape(str(value))
    return SPACE_RE.sub(" ", TAG_RE.sub(" ", text)).strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKD", value.lower())
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return SPACE_RE.sub(" ", value).strip()


RESOURCE_PATTERNS = {
    "medical": re.compile(
        r"\b(?:sacyl|emergencias sanitarias|ambulancia|uvi|u\.?m\.?e\.?|"
        r"soporte vital|personal sanitari[oa]|centro de salud|punto de atenci[oó]n continuada|"
        r"helic[oó]ptero medicalizado|m[eé]dico)\b", re.I
    ),
    "police_security": re.compile(
        r"\b(?:guardia civil|polic[ií]a|cuerpo nacional|agentes? de la autoridad)\b", re.I
    ),
    "fire_rescue": re.compile(
        r"\b(?:bomberos?|extinci[oó]n de incendios|servicio de salvamento|grupo de rescate|"
        r"rescate y salvamento|helic[oó]ptero de rescate)\b", re.I
    ),
    "traffic_road": re.compile(
        r"\b(?:tr[aá]fico|carreteras?|fomento|mantenimiento de carreteras|concesionaria|gr[uú]a)\b", re.I
    ),
    "civil_environment": re.compile(
        r"\b(?:protecci[oó]n civil|medioambient\w*|agentes? forestales|confederaci[oó]n hidrogr[aá]fica|"
        r"junta de castilla y le[oó]n|medio ambiente|servicio territorial)\b", re.I
    ),
}

SERVICE_ENTITY_RE = re.compile(
    r"\b(?:guardia civil(?: de tr[aá]fico)?|polic[ií]a(?: local| municipal| nacional)?|"
    r"cuerpo nacional de polic[ií]a|bomberos?|"
    r"emergencias sanitarias(?:\s*-?\s*sacyl)?|sacyl|ambulancia(?: de soporte vital b[aá]sico)?|"
    r"uvi m[oó]vil|unidad medicalizada de emergencias|helic[oó]ptero medicalizado|"
    r"punto de atenci[oó]n continuada|centro coordinador de urgencias|"
    r"servicio de extinci[oó]n de incendios y salvamento|grupo de rescate|protecci[oó]n civil|"
    r"agentes? forestales|personal sanitari[oa])\b",
    re.I,
)

TARGET_SERVICE_TOKEN = (
    r"(?:\[SERVICIO\]|guardia civil|polic[ií]a|cuerpo nacional de polic[ií]a|bomberos?|"
    r"emergencias sanitarias|sacyl|ambulancia|uvi m[oó]vil|unidad medicalizada|"
    r"centro coordinador de urgencias|protecci[oó]n civil|grupo de rescate|"
    r"personal sanitari[oa]|agentes? forestales|subsector de tr[aá]fico)"
)

CALL_START_RE = re.compile(
    r"\b(?:(?:el\s+)?(?:servicio|centro) de emergencias.{0,120}?(?:atiende|atendi[oó]|recibe|recibi[oó])\s+(?:una|la)\s+llamada|"
    r"la sala de operaciones.{0,100}?(?:atiende|atendi[oó]|recibe|recibi[oó])\s+(?:una|la)\s+llamada|"
    r"tras recibirse.{0,100}?una llamada|(?:fue\s+)?(?:el|la)\s+prop\w+.{0,80}?quien\s+contact[oó]\s+con|"
    r"(?:el|la)\s+prop\w+.{0,80}?(?:ha|hab[ií]a)\s+contactado\s+(?:a|con)|"
    r"quien\s+contact[oó]\s+con|(?:ha|hab[ií]a)\s+contactado\s+(?:a|con)|contact[oó]\s+con\s+(?:el|la)|"
    r"se reciben?|reciben?|una llamada|varias llamadas|dos llamadas|numerosas llamadas|"
    r"un alertante|los alertantes|la persona que llama)\b",
    re.I,
)

BOUNDARY_RULES = [
    (
        "operator_multiconference",
        re.compile(
            r"\b(?:el\s+)?(?:gestor|operador)(?:a)?\s+del\s+(?:1-1-2|112)\b.{0,100}?"
            r"\b(?:realiza|hace)\b.{0,60}?\b(?:multi)?conferencia\b",
            re.I,
        ),
    ),
    (
        "any_dispatch_notice",
        re.compile(r"\b(?:da|dio|dieron|ha dado|hab[ií]a dado|pasa|pas[oó]|traslada|comunica|cursa)\s+(?:el\s+)?aviso\b", re.I),
    ),
    (
        "incident_communicated",
        re.compile(r"\b(?:se comunica|comunicada)\s+(?:la\s+)?incidencia\b", re.I),
    ),
    (
        "dispatch_notice",
        re.compile(
            r"\b(?:la sala de operaciones|el centro de emergencias|desde el centro de emergencias|"
            r"el 1-1-2|el gestor del 1-1-2|desde la sala)\b.{0,120}?"
            r"\b(?:da|dio|dieron|ha dado|hab[ií]a dado|pasa|pas[oó]|traslada|comunica|cursa)\s+(?:el\s+)?aviso\b",
            re.I,
        ),
    ),
    (
        "generic_dispatch_notice",
        re.compile(
            r"\b(?:se|tambi[eé]n se)\s+(?:da|dio|dieron|ha dado|hab[ií]a dado|pasa|pas[oó]|traslada|comunica|cursa)\s+(?:el\s+)?aviso\b",
            re.I,
        ),
    ),
    (
        "dispatch_information_to_service",
        re.compile(
            rf"\b(?:desde\s+(?:la\s+sala\s+del\s+)?(?:1-1-2|112)|"
            rf"el\s+(?:1-1-2|112|centro\s+de\s+emergencias)|"
            rf"la\s+sala(?:\s+de\s+operaciones|\s+del\s+(?:1-1-2|112))?|"
            rf"el\s+centro\s+de\s+emergencias|desde\s+la\s+sala)\b.{{0,140}}?"
            rf"\b(?:se\s+)?(?:informa|avisa|alerta|comunica|ha\s+informado|ha\s+avisado|"
            rf"ha\s+alertado|ha\s+comunicado)\b.{{0,120}}?"
            rf"\ba(?:l|\s+los|\s+las)?\s+.{{0,70}}?\b{TARGET_SERVICE_TOKEN}\b",
            re.I,
        ),
    ),
    (
        "dispatch_notice_112_variant",
        re.compile(
            r"\b(?:la sala (?:del|de operaciones del?)?\s*(?:1-1-2|112)|el\s+(?:1-1-2|112))\s+"
            r"(?:da|dio|dieron|ha dado|hab[ií]a dado|pasa|pas[oó]|traslada|comunica|cursa)\s+(?:el\s+)?aviso\b",
            re.I,
        ),
    ),
    ("resources_dispatched", re.compile(r"\b(?:se|que)\s+(?:moviliza|movilizan|env[ií]a|env[ií]an|activa|activan|desplaza|desplazan)\b", re.I)),
    (
        "service_dispatches_resources",
        re.compile(
            rf"\b{TARGET_SERVICE_TOKEN}\b.{{0,80}}?\b(?:moviliza|movilizan|env[ií]a|env[ií]an|"
            rf"activa|activan|desplaza|desplazan)\b",
            re.I,
        ),
    ),
    ("call_transferred", re.compile(r"\bse\s+transfiere\s+la\s+llamada\b", re.I)),
    (
        "direct_service_activation",
        re.compile(
            r"\bse\s+avisa\s+a(?:l|los|las)?\s+(?:emergencias sanitarias|sacyl|guardia civil|"
            r"polic[ií]a|bomberos?|ambulancia|protecci[oó]n civil)\b",
            re.I,
        ),
    ),
    ("on_scene", re.compile(r"\b(?:en el lugar|una vez en el lugar|a su llegada|al llegar)\b", re.I)),
    ("response_services", re.compile(r"\b(?:los|las)\s+(?:servicios|organismos|equipos|efectivos)\s+de\s+emergencia\b", re.I)),
    ("intervention", re.compile(r"\b(?:intervienen|interviniendo|acuden|se personan|socorren|rescatan|evac[uú]an|trasladan)\b", re.I)),
    (
        "completed_response_action",
        re.compile(
            r"\b(?:rescataron|socorrieron|evacuaron|trasladaron|intervinieron|acudieron|"
            r"se personaron|trabajaron|extinguieron|sofocaron|auxiliaron|"
            r"(?:ha|han|hab[ií]a|hab[ií]an)\s+acudido|"
            r"(?:ha|han|hab[ií]a|hab[ií]an)\s+(?:rescatado|socorrido|evacuado|trasladado|atendido|auxiliado)|"
            r"(?:ha|han)\s+tenido\s+que\s+ser\s+(?:rescatad[oa]s?|socorrid[oa]s?|evacuad[oa]s?|trasladad[oa]s?|atendid[oa]s?|auxiliad[oa]s?)|"
            r"(?:fue|fueron|ha sido|han sido|hab[ií]a sido|hab[ií]an sido)\s+"
            r"(?:rescatad[oa]s?|socorrid[oa]s?|evacuad[oa]s?|trasladad[oa]s?|atendid[oa]s?|auxiliad[oa]s?))\b",
            re.I,
        ),
    ),
    (
        "active_response_status",
        re.compile(
            r"\b(?:ya\s+)?se\s+encuentra(?:n)?\s+(?:trabajando|actuando|atendiendo|interviniendo)|"
            r"\bse\s+encuentra(?:n)?\s+ya\s+(?:trabajando|actuando|atendiendo|interviniendo)|"
            r"\b(?:que\s+)?acude(?:n)?\s+solicita(?:n)?\b|\best[aá]n\s+actuando\b|"
            r"\bseg[uú]n\s+confirma(?:n)?\b|\batendid[oa]s?\s+por\b|\blabores\s+de\s+extinci[oó]n\b",
            re.I,
        ),
    ),
    (
        "responders_already_active",
        re.compile(
            rf"\b{TARGET_SERVICE_TOKEN}\b.{{0,180}}?\b(?:ha\s+acudido|han\s+acudido|"
            rf"se\s+ha\s+desplazado|se\s+han\s+desplazado|se\s+encuentra(?:n)?.{{0,100}}?"
            rf"(?:atendiendo|actuando|trabajando|interviniendo)|atiende(?:n)?|trabaja(?:n)?)\b",
            re.I,
        ),
    ),
    (
        "intervention_summary",
        re.compile(
            r"\b(?:en|durante)\s+la\s+intervenci[oó]n\b|"
            r"\bel\s+operativo\s+de\s+(?:b[uú]squeda|rescate|emergencia)\b",
            re.I,
        ),
    ),
]

PREAMBLE_DOWNSTREAM_RE = re.compile(
    r"\b(?:ha dirigido|ha confirmado|confirmad[oa]s?|socorr\w*|rescat\w*|auxili\w*|"
    r"atendid[oa]s?|evacuad[oa]s?|trasladad[oa]s?|intervin\w*|trabaj\w*|"
    r"intervenci[oó]n de|operativo de (?:rescate|b[uú]squeda)|"
    r"(?:fue|fueron|ha sido|han sido)\s+(?:atendid|rescatad|evacuad|trasladad|auxiliad))\w*\b",
    re.I,
)

DOWNSTREAM_AUDIT_RE = re.compile(
    rf"\b(?:(?:el\s+)?(?:gestor|operador)(?:a)?\s+del\s+(?:1-1-2|112).{{0,100}}?"
    rf"(?:realiza|hace).{{0,60}}?(?:multi)?conferencia|"
    rf"primera valoraci[oó]n|hasta que lleguen|"
    rf"da aviso|dio aviso|ha dado aviso|hab[ií]a dado aviso|pasa aviso|se comunica la incidencia|comunicada la incidencia|"
    rf"(?:el|desde el|la sala del|desde la sala del|el centro de emergencias|la sala de operaciones del?)\s*"
    rf"(?:1-1-2|112)?.{{0,140}}?(?:informa|avisa|alerta|comunica).{{0,120}}?"
    rf"a(?:l|\s+los|\s+las)?\s+.{{0,70}}?{TARGET_SERVICE_TOKEN}|se transfiere la llamada|"
    rf"se moviliza|se movilizan|se env[ií]a|se env[ií]an|{TARGET_SERVICE_TOKEN}.{{0,80}}?"
    rf"(?:moviliza|movilizan|env[ií]a|env[ií]an|activa|activan|desplaza|desplazan)|"
    r"se activa|se desplaza|en el lugar|intervienen|acuden|se personan|"
    r"los servicios de emergencia|los organismos de emergencia|rescataron|socorrieron|evacuaron|"
    r"trasladaron|intervinieron|trabajaron|extinguieron|sofocaron|auxiliaron|(?:ha|han|hab[ií]a|hab[ií]an) acudido|"
    r"(?:ha|han|hab[ií]a|hab[ií]an) (?:rescatado|socorrido|evacuado|trasladado|atendido|auxiliado)|"
    r"(?:ha|han) tenido que ser (?:rescatad[oa]s?|socorrid[oa]s?|evacuad[oa]s?|trasladad[oa]s?|atendid[oa]s?|auxiliad[oa]s?)|"
    r"(?:fue|fueron|ha sido|han sido)\s+(?:rescatad[oa]s?|socorrid[oa]s?|evacuad[oa]s?|"
    r"trasladad[oa]s?|atendid[oa]s?|auxiliad[oa]s?)|(?:en|durante) la intervenci[oó]n|"
    r"el operativo de (?:b[uú]squeda|rescate|emergencia)|"
    r"(?:ya\s+)?se encuentra(?:n)? (?:trabajando|actuando|atendiendo|interviniendo)|"
    r"se encuentra(?:n)? ya (?:trabajando|actuando|atendiendo|interviniendo)|"
    r"(?:que )?acude(?:n)? solicita(?:n)?|est[aá]n actuando|seg[uú]n confirma(?:n)?|"
    rf"atendid[oa]s? por|labores de extinci[oó]n|{TARGET_SERVICE_TOKEN}.{{0,180}}?"
    rf"(?:ha acudido|han acudido|se ha desplazado|se han desplazado|se encuentra(?:n)?.{{0,100}}?"
    rf"(?:atendiendo|actuando|trabajando|interviniendo)))\b",
    re.I,
)

MULTI_INCIDENT_SUMMARY_RE = re.compile(
    r"\b(?:dos|tres|cuatro|cinco|seis|siete|varios|m[uú]ltiples)\s+accidentes\b|"
    r"\ba\s+lo\s+largo\s+de\s+la\s+jornada\b.{0,160}\baccidentes\b",
    re.I,
)


def construct_predispatch(narrative: str) -> dict[str, object]:
    original = clean_html(narrative)
    start = 0
    call = CALL_START_RE.search(original)
    if call and call.start() > 0 and PREAMBLE_DOWNSTREAM_RE.search(original[: call.start()]):
        start = call.start()
    candidate = original[start:]
    boundary_name = "none"
    boundary_position = len(candidate)
    for name, pattern in BOUNDARY_RULES:
        match = pattern.search(candidate)
        if not match:
            continue
        match_position = match.start()
        if name in {
            "completed_response_action",
            "active_response_status",
            "responders_already_active",
            "service_dispatches_resources",
            "intervention_summary",
        }:
            previous_period = candidate.rfind(". ", 0, match_position)
            previous_semicolon = candidate.rfind("; ", 0, match_position)
            previous_boundary = max(previous_period, previous_semicolon)
            match_position = previous_boundary + 2 if previous_boundary >= 0 else 0
        if match_position < boundary_position:
            boundary_name = name
            boundary_position = match_position
    truncated = candidate[:boundary_position].strip(" .;,:-")
    truncated = re.sub(
        r"\s+\b(?:desde|posteriormente|tambi[eé]n|el|la|los|las|y)\s*$",
        "",
        truncated,
        flags=re.I,
    ).strip(" .;,:-")
    if len(truncated) < 40 and boundary_name == "none":
        truncated = candidate[: min(len(candidate), 500)].strip(" .;,:-")
        boundary_name = f"fallback_after_{boundary_name}"
    entity_count = len(SERVICE_ENTITY_RE.findall(truncated))
    redacted = SERVICE_ENTITY_RE.sub(" [SERVICIO] ", truncated)
    redacted = SPACE_RE.sub(" ", redacted).strip()
    return {
        "predispatch_text": redacted,
        "boundary_rule": boundary_name,
        "start_offset": start,
        "end_offset": start + boundary_position,
        "characters_removed": max(0, len(original) - len(truncated)),
        "service_entities_redacted": entity_count,
        "modified": int(redacted != original),
        "original_downstream_matches": len(DOWNSTREAM_AUDIT_RE.findall(original)),
        "remaining_downstream_matches": len(DOWNSTREAM_AUDIT_RE.findall(redacted)),
    }


OLD_FACT_PATTERNS = {
    "injury": re.compile(r"herid[oa]s?|lesionad[oa]s?|traumatismo|quemadur|sangr", re.I),
    "unconscious": re.compile(r"inconsciente|no responde|sin conocimiento|desvanecid", re.I),
    "trapped": re.compile(r"atrapad[oa]s?|encarcelad[oa]s?|no puede salir|aprisionad", re.I),
    "fire_smoke": re.compile(r"incendio|fuego|llamas|humo|arde|quem[aá]ndose", re.I),
    "hazardous_leak": re.compile(r"fuga de gas|escape de gas|derrame|sustancia peligrosa|t[oó]xic", re.I),
    "violence_weapon": re.compile(r"agresi[oó]n|arma|apuñal|disparo|violencia|pelea", re.I),
    "water_drowning": re.compile(r"ahog|sumergid|r[ií]o|embalse|piscina|agua", re.I),
    "missing_person": re.compile(r"desaparecid|no localizad|b[uú]squeda de una persona", re.I),
    "multiple_people": re.compile(r"varias personas|m[uú]ltiples|\b[2-9]\s+(?:personas|heridos|ocupantes|pacientes)|dos personas|tres personas|cuatro personas", re.I),
}

PERSON_WATER = (
    r"(?:persona|niñ[oa]|menor|var[oó]n|mujer|bañista|nadador|cuerpo|ocupante)"
)
WATER_CONTEXT = r"(?:r[ií]o|embalse|piscina|canal|pozo|agua|lago|pantano)"

FACT_PATTERNS = {
    "injury": re.compile(r"\b(?:herid[oa]s?|lesionad[oa]s?|traumatismo|quemaduras?|sangrado|sangra)\b", re.I),
    "unconscious": re.compile(r"\b(?:inconsciente|no responde|sin conocimiento|desvanecid[oa])\b", re.I),
    "trapped": re.compile(r"\b(?:atrapad[oa]s?|encarcelad[oa]s?|aprisionad[oa]s?|no puede salir)\b", re.I),
    "fire_smoke": re.compile(r"\b(?:incendio|fuego|llamas|humo|arde|ardiendo|quem[aá]ndose)\b", re.I),
    "hazardous_leak": re.compile(r"\b(?:fuga de gas|escape de gas|derrame|sustancia peligrosa|t[oó]xic[oa]s?)\b", re.I),
    "violence_weapon": re.compile(r"\b(?:agresi[oó]n|arma|apuñal\w*|disparo|violencia|pelea)\b", re.I),
    "water_drowning": re.compile(
        rf"\b(?:ahogad[oa]s?|ahogamiento|sumergid[oa]s?)\b|"
        rf"\b{PERSON_WATER}\b.{{0,55}}\b{WATER_CONTEXT}\b|"
        rf"\b{WATER_CONTEXT}\b.{{0,55}}\b{PERSON_WATER}\b",
        re.I,
    ),
    "missing_person": re.compile(r"\b(?:desaparecid[oa]s?|no localizad[oa]s?|b[uú]squeda de una persona)\b", re.I),
    "multiple_people": re.compile(
        r"\b(?:varias personas|m[uú]ltiples|[2-9]\s+(?:personas|heridos|ocupantes|pacientes)|"
        r"dos personas|tres personas|cuatro personas)\b",
        re.I,
    ),
}

TYPE_PATTERNS = [
    ("fire", re.compile(r"\b(?:incendio|fuego|llamas|humo|explosi[oó]n|arde|ardiendo)\b", re.I)),
    ("traffic_accident", re.compile(r"\b(?:accidente (?:de )?tr[aá]fico|colisi[oó]n|atropell\w*|vuelco|salida de v[ií]a|carretera|autov[ií]a|turismo|cam[ií]on|motocicleta|motorista)\b", re.I)),
    ("rescue", re.compile(r"\b(?:rescate|senderista|montañ\w*|barranco|cueva|precipitad[oa]|desaparecid[oa]|atrapad[oa]|ahogad[oa]|embalse|r[ií]o)\b", re.I)),
    ("security", re.compile(r"\b(?:agresi[oó]n|pelea|arma|apuñal\w*|disparo|robo|amenaza|violencia)\b", re.I)),
    ("environment_weather_leak", re.compile(r"\b(?:inundaci[oó]n|tormenta|viento|nieve|meteor\w*|fuga|escape de gas|derrame|contaminaci[oó]n|vertido)\b", re.I)),
    ("medical", re.compile(r"\b(?:inconsciente|enfermedad|intoxicaci[oó]n|asistencia sanitaria|parada card\w*|desvanecid[oa]|dolor|parto|herid[oa])\b", re.I)),
]

S2_RE = re.compile(
    r"\b(?:fallecid[oa]s?|muert[oa]s?|sin vida|inconsciente|no respira|parada card\w*|"
    r"atrapad[oa]s?|ahogad[oa]s?|ahogamiento|explosi[oó]n|arma|apuñal\w*|disparo|"
    r"herid[oa] grave|heridas graves|varias personas|m[uú]ltiples v[ií]ctimas)\b",
    re.I,
)
S0_RE = re.compile(
    r"\b(?:obst[aá]culo|animal suelto|rama ca[ií]da|señal ca[ií]da|incidencia de tr[aá]fico|"
    r"sin heridos|daños materiales|falsa alarma|sin riesgo)\b",
    re.I,
)
AMBIG_RE = re.compile(
    r"\b(?:no (?:puede|pueden) confirmar|sin confirmar|se desconoce|no se sabe|posible|"
    r"parece|no responde|no localizad[oa])\b",
    re.I,
)


def derive_type(text: str) -> str:
    for label, pattern in TYPE_PATTERNS:
        if pattern.search(text):
            return label
    return "other"


def derive_severity(text: str) -> str:
    if S2_RE.search(text):
        return "S2"
    if S0_RE.search(text):
        return "S0"
    return "S1"


def required_information(category: str) -> list[str]:
    return {
        "fire": ["people_at_risk", "fire_or_smoke_status", "exact_location", "hazardous_material"],
        "traffic_accident": ["injury_status", "trapped_status", "vehicle_count", "exact_location"],
        "medical": ["consciousness", "breathing", "symptoms", "exact_location"],
        "security": ["ongoing_threat", "weapon_status", "injury_status", "exact_location"],
        "rescue": ["people_at_risk", "access_conditions", "injury_status", "exact_location"],
        "environment_weather_leak": ["substance_or_hazard", "exposure", "spread", "exact_location"],
        "other": ["incident_nature", "people_at_risk", "exact_location"],
    }[category]


INFO_EVIDENCE = {
    "people_at_risk": re.compile(r"\b(?:persona|ocupante|vecin[oa]|trabajador|menor)\b", re.I),
    "fire_or_smoke_status": FACT_PATTERNS["fire_smoke"],
    "exact_location": re.compile(r"\b(?:calle|kil[oó]metro|carretera|localidad|municipio|vivienda)\b", re.I),
    "hazardous_material": FACT_PATTERNS["hazardous_leak"],
    "injury_status": re.compile(r"\b(?:herid\w*|iles[oa]|sin heridos|lesion\w*)\b", re.I),
    "trapped_status": re.compile(r"\b(?:atrapad\w*|no atrapad\w*|puede salir|encarcelad\w*)\b", re.I),
    "vehicle_count": re.compile(r"\b(?:veh[ií]culo|turismo|cam[ií]on|moto|autob[uú]s)\b", re.I),
    "consciousness": re.compile(r"\b(?:consciente|inconsciente|responde|sin conocimiento)\b", re.I),
    "breathing": re.compile(r"\b(?:respira|no respira|respiraci[oó]n)\b", re.I),
    "symptoms": re.compile(r"\b(?:dolor|mareo|s[ií]ntoma|convulsi\w*|sangr\w*|trauma\w*)\b", re.I),
    "ongoing_threat": re.compile(r"\b(?:en curso|ha huido|contin[uú]a|amenaza|agresi[oó]n)\b", re.I),
    "weapon_status": re.compile(r"\b(?:arma|sin armas|cuchillo|pistola|escopeta)\b", re.I),
    "access_conditions": re.compile(r"\b(?:acceso|sendero|montaña|barranco|helic[oó]ptero)\b", re.I),
    "substance_or_hazard": re.compile(r"\b(?:gas|combustible|qu[ií]mic\w*|humo|agua|viento|nieve)\b", re.I),
    "exposure": re.compile(r"\b(?:expuest\w*|afectad\w*|intoxicad\w*|inhalaci[oó]n)\b", re.I),
    "spread": re.compile(r"\b(?:extiende|propaga|afecta a|inunda)\b", re.I),
    "incident_nature": re.compile(r"\b(?:incidente|accidente|problema|aviso)\b", re.I),
}


def load_and_repair() -> pd.DataFrame:
    frames = []
    for filename, encoding, source in [
        ("jcyl_emergencias_2008_2022.csv", "latin1", "JCYL_2008_2022"),
        ("jcyl_emergencias_2023_actualidad.csv", "utf-8-sig", "JCYL_2023_CURRENT"),
    ]:
        frame = pd.read_csv(RAW / filename, sep=";", encoding=encoding, low_memory=False)
        frame["source"] = source
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True, sort=False)
    df["date_num"] = pd.to_numeric(df["FechaIncidente"], errors="coerce")
    df = df[df["date_num"].between(20080101, 20251231)].copy()
    df["date"] = pd.to_datetime(df["date_num"].astype("int64").astype(str), format="%Y%m%d")
    df["year"] = df["date"].dt.year
    df["seed_incident_id"] = df["Enlace al contenido"].astype(str).str.extract(r"/Incidente/(\d+)/", expand=False)
    df["title"] = df["Título"].map(clean_html)
    df["narrative_original"] = df["DescripcionBlob"].map(clean_html)
    df["resource_text"] = df["MediosMov"].map(clean_html)
    df["multi_incident_summary"] = df["narrative_original"].map(
        lambda text: int(bool(MULTI_INCIDENT_SUMMARY_RE.search(text)))
    )
    repaired = pd.DataFrame(df["narrative_original"].map(construct_predispatch).tolist(), index=df.index)
    df = pd.concat([df, repaired], axis=1)

    for label, pattern in RESOURCE_PATTERNS.items():
        df[f"reference_{label}"] = df["resource_text"].map(lambda x: int(bool(pattern.search(x))))
    df["observed_reference_activation_set"] = df.apply(
        lambda row: [label for label in RESOURCE_LABELS if row[f"reference_{label}"] == 1], axis=1
    )
    df["incident_type"] = df["predispatch_text"].map(derive_type)
    df["severity"] = df["predispatch_text"].map(derive_severity)
    for fact in FACT_LABELS:
        df[f"fact_old_original_{fact}"] = df["narrative_original"].map(lambda x: int(bool(OLD_FACT_PATTERNS[fact].search(x))))
        df[f"fact_old_predispatch_{fact}"] = df["predispatch_text"].map(lambda x: int(bool(OLD_FACT_PATTERNS[fact].search(x))))
        df[f"fact_{fact}"] = df["predispatch_text"].map(lambda x: int(bool(FACT_PATTERNS[fact].search(x))))
    df["material_facts"] = df.apply(lambda row: [fact for fact in FACT_LABELS if row[f"fact_{fact}"]], axis=1)
    df["required_information"] = df["incident_type"].map(required_information)
    df["missing_information_reference"] = df.apply(
        lambda row: [field for field in row["required_information"] if not INFO_EVIDENCE[field].search(row["predispatch_text"])],
        axis=1,
    )
    df["ambiguous"] = df.apply(
        lambda row: int(bool(AMBIG_RE.search(row["predispatch_text"])) or len(row["missing_information_reference"]) >= 2),
        axis=1,
    )
    df["reversibility"] = df["severity"].map({"S0": "REV0", "S1": "REV1", "S2": "REV2"})
    df["protected_predecision"] = (
        df["severity"].eq("S2") | df["ambiguous"].astype(bool) | df["incident_type"].isin(["fire", "rescue"])
    ).astype(int)
    eligible = (
        df["predispatch_text"].str.len().ge(60)
        & df["resource_text"].str.len().gt(0)
        & df["observed_reference_activation_set"].map(len).gt(0)
        & df["seed_incident_id"].notna()
        & ~df["multi_incident_summary"].astype(bool)
    )
    df["eligible_before_duplicate_audit"] = eligible.astype(int)
    eligibility_audit = {
        "valid_dated_rows_considered": int(len(df)),
        "excluded_multi_incident_summaries": int(df["multi_incident_summary"].sum()),
        "rows_with_predispatch_text_under_60_characters": int(df["predispatch_text"].str.len().lt(60).sum()),
        "rows_without_nonempty_observed_reference_activation_set": int(
            df["observed_reference_activation_set"].map(len).eq(0).sum()
        ),
        "eligible_before_duplicate_audit": int(eligible.sum()),
        "exclusion_counts_are_nonexclusive": True,
    }
    (STATISTICS / "eligibility_audit.json").write_text(
        json.dumps(eligibility_audit, indent=2), encoding="utf-8"
    )
    return df.loc[eligible].reset_index(drop=True)


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def duplicate_audit(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    work = df.copy()
    work["normalized_predispatch"] = work["predispatch_text"].map(normalize_text)
    work["exact_text_hash"] = work["normalized_predispatch"].map(lambda x: hashlib.sha256(x.encode()).hexdigest())
    work = work.sort_values(["date", "seed_incident_id"]).reset_index(drop=True)
    exact_duplicate = work.duplicated("exact_text_hash", keep="first")
    exact_pairs = work.loc[exact_duplicate, ["seed_incident_id", "date", "exact_text_hash"]].copy()
    exact_pairs["audit_type"] = "exact_duplicate_removed"
    retained = work.loc[~exact_duplicate].reset_index(drop=True)

    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=45000, sublinear_tf=True)
    matrix = vectorizer.fit_transform(retained["normalized_predispatch"])
    neighbors = NearestNeighbors(n_neighbors=min(4, len(retained)), metric="cosine", algorithm="brute", n_jobs=-1)
    neighbors.fit(matrix)
    distances, indices = neighbors.kneighbors(matrix)
    uf = UnionFind(len(retained))
    near_pair_rows = []
    for i in range(len(retained)):
        for distance, j in zip(distances[i, 1:], indices[i, 1:]):
            similarity = 1.0 - float(distance)
            day_gap = abs((retained.loc[i, "date"] - retained.loc[j, "date"]).days)
            if similarity >= 0.94 and day_gap <= 14:
                uf.union(i, int(j))
                near_pair_rows.append({
                    "audit_type": "near_duplicate_same_incident",
                    "left_id": retained.loc[i, "seed_incident_id"],
                    "right_id": retained.loc[j, "seed_incident_id"],
                    "left_date": retained.loc[i, "date"],
                    "right_date": retained.loc[j, "date"],
                    "similarity": similarity,
                    "day_gap": day_gap,
                    "left_text": retained.loc[i, "predispatch_text"],
                    "right_text": retained.loc[j, "predispatch_text"],
                })
    retained["near_group"] = [uf.find(i) for i in range(len(retained))]
    retained["near_group_size"] = retained.groupby("near_group")["near_group"].transform("size")
    retained["near_duplicate_removed"] = retained.duplicated("near_group", keep="first")
    near_removed_count = int(retained["near_duplicate_removed"].sum())
    retained = retained.loc[~retained["near_duplicate_removed"]].reset_index(drop=True)

    # Strict cross-period audit: later cases nearly identical to any earlier-period case
    # are excluded from validation/test to prevent templated narrative contamination.
    retained["split_period"] = np.select(
        [retained["year"].le(2022), retained["year"].eq(2023), retained["year"].isin([2024, 2025])],
        ["train", "validation", "test_pool"],
        default="excluded_date",
    )
    contamination_rows = []
    contaminated_ids: set[str] = set()
    for later_period, earlier_mask in [
        ("validation", retained["split_period"].eq("train")),
        ("test_pool", retained["split_period"].isin(["train", "validation"])),
    ]:
        later_mask = retained["split_period"].eq(later_period)
        if not later_mask.any() or not earlier_mask.any():
            continue
        compare_vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=45000, sublinear_tf=True)
        compare_vectorizer.fit(pd.concat([retained.loc[earlier_mask, "normalized_predispatch"], retained.loc[later_mask, "normalized_predispatch"]]))
        x_early = compare_vectorizer.transform(retained.loc[earlier_mask, "normalized_predispatch"])
        x_late = compare_vectorizer.transform(retained.loc[later_mask, "normalized_predispatch"])
        nn = NearestNeighbors(n_neighbors=1, metric="cosine", algorithm="brute", n_jobs=-1).fit(x_early)
        dist, ind = nn.kneighbors(x_late)
        early_rows = retained.index[earlier_mask].to_numpy()
        late_rows = retained.index[later_mask].to_numpy()
        for k, later_idx in enumerate(late_rows):
            similarity = 1.0 - float(dist[k, 0])
            if similarity >= 0.95:
                earlier_idx = early_rows[int(ind[k, 0])]
                later_id = str(retained.loc[later_idx, "seed_incident_id"])
                contaminated_ids.add(later_id)
                contamination_rows.append({
                    "audit_type": "cross_period_near_duplicate_removed",
                    "left_id": retained.loc[earlier_idx, "seed_incident_id"],
                    "right_id": later_id,
                    "left_date": retained.loc[earlier_idx, "date"],
                    "right_date": retained.loc[later_idx, "date"],
                    "similarity": similarity,
                    "day_gap": abs((retained.loc[later_idx, "date"] - retained.loc[earlier_idx, "date"]).days),
                    "left_text": retained.loc[earlier_idx, "predispatch_text"],
                    "right_text": retained.loc[later_idx, "predispatch_text"],
                })
    retained["cross_period_contaminated"] = retained["seed_incident_id"].astype(str).isin(contaminated_ids)
    final = retained.loc[~retained["cross_period_contaminated"]].reset_index(drop=True)
    pairs = pd.concat([pd.DataFrame(near_pair_rows), pd.DataFrame(contamination_rows)], ignore_index=True, sort=False)
    audit = {
        "eligible_before_duplicate_audit": int(len(df)),
        "exact_duplicate_narratives_removed": int(exact_duplicate.sum()),
        "near_duplicate_same_incident_removed": near_removed_count,
        "cross_period_near_duplicates_removed": int(len(contaminated_ids)),
        "eligible_after_duplicate_audit": int(len(final)),
        "remaining_duplicate_seed_ids": int(final["seed_incident_id"].duplicated().sum()),
        "near_duplicate_similarity_threshold": 0.94,
        "same_incident_max_day_gap": 14,
        "cross_period_contamination_threshold": 0.95,
    }
    if len(exact_pairs):
        exact_pairs.to_csv(STATISTICS / "exact_duplicate_rows.csv", index=False)
    pairs.to_csv(STATISTICS / "near_duplicate_audit_pairs.csv", index=False)
    return final, pairs, audit


def choose_thresholds(y_true: np.ndarray, probabilities: np.ndarray) -> list[float]:
    grid = np.arange(0.10, 0.81, 0.05)
    thresholds = []
    for column in range(y_true.shape[1]):
        scores = [f1_score(y_true[:, column], probabilities[:, column] >= threshold, zero_division=0) for threshold in grid]
        thresholds.append(float(grid[int(np.argmax(scores))]))
    return thresholds


def expected_calibration_error(y: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    result = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (probability >= low) & (probability < high if high < 1 else probability <= high)
        if mask.any():
            result += mask.mean() * abs(y[mask].mean() - probability[mask].mean())
    return float(result)


def top_lexical_terms(matrix: sparse.spmatrix, names: np.ndarray, model: OneVsRestClassifier, predicted: np.ndarray, row_index: int, limit: int = 5) -> list[str]:
    row = matrix.getrow(row_index)
    scores: Counter[str] = Counter()
    for label_idx in np.where(predicted[row_index] == 1)[0]:
        contributions = row.multiply(model.estimators_[label_idx].coef_).tocoo()
        order = np.argsort(contributions.data)[-limit:]
        for position in order:
            if contributions.data[position] > 0:
                scores[str(names[contributions.col[position]])] += float(contributions.data[position])
    return [term for term, _ in scores.most_common(limit)]


def metric_values(y_true: np.ndarray, y_pred: np.ndarray, exact_probability: np.ndarray) -> dict[str, float]:
    exact = np.all(y_true == y_pred, axis=1).astype(int)
    return {
        "exact_activation_set_accuracy": float(exact.mean()),
        "micro_precision": float(precision_score(y_true, y_pred, average="micro", zero_division=0)),
        "micro_recall": float(recall_score(y_true, y_pred, average="micro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confidence_brier": float(brier_score_loss(exact, exact_probability)),
        "confidence_ece_10": expected_calibration_error(exact, exact_probability, 10),
        "confidence_log_loss": float(log_loss(exact, np.column_stack([1 - exact_probability, exact_probability]), labels=[0, 1])),
    }


def bootstrap_ai_metrics(y_true: np.ndarray, y_pred: np.ndarray, confidence: np.ndarray, replications: int = 1000) -> tuple[dict, pd.DataFrame]:
    point = metric_values(y_true, y_pred, confidence)
    rng = np.random.default_rng(GLOBAL_SEED + 11)
    rows = []
    for replication in range(replications):
        indices = rng.integers(0, len(y_true), len(y_true))
        row = {"replication": replication}
        row.update(metric_values(y_true[indices], y_pred[indices], confidence[indices]))
        rows.append(row)
    boot = pd.DataFrame(rows)
    result = {}
    for metric, value in point.items():
        result[metric] = {
            "estimate": value,
            "ci_low": float(boot[metric].quantile(0.025)),
            "ci_high": float(boot[metric].quantile(0.975)),
            "bootstrap_replications": replications,
        }
    return result, boot


def train_and_freeze(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    train = df[df["year"].le(2022)].copy()
    validation = df[df["year"].eq(2023)].copy()
    test_pool = df[df["year"].isin([2024, 2025])].copy()
    if len(test_pool) < 1200:
        raise RuntimeError(f"Only {len(test_pool)} leakage-free 2024-2025 records; 1,200 required")
    splitter = StratifiedShuffleSplit(n_splits=1, train_size=1200, random_state=GLOBAL_SEED)
    selected, reserve = next(splitter.split(test_pool, test_pool["incident_type"]))
    test = test_pool.iloc[selected].sort_values(["date", "seed_incident_id"]).reset_index(drop=True)
    reserve_frame = test_pool.iloc[reserve].copy()

    split_rows = []
    for name, frame in [("train", train), ("validation", validation), ("test", test), ("test_reserve", reserve_frame)]:
        split_rows.extend({"seed_incident_id": row.seed_incident_id, "split": name, "year": row.year} for row in frame.itertuples())
    pd.DataFrame(split_rows).to_csv(PROCESSED / "frozen_split_assignments.csv", index=False)

    vectorizer = FeatureUnion([
        ("word", TfidfVectorizer(lowercase=True, strip_accents="unicode", ngram_range=(1, 2), min_df=2, max_df=0.98, max_features=30000, sublinear_tf=True)),
        ("char", TfidfVectorizer(lowercase=True, strip_accents="unicode", analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=25000, sublinear_tf=True)),
    ])
    x_train = vectorizer.fit_transform(train["predispatch_text"])
    x_validation = vectorizer.transform(validation["predispatch_text"])
    x_test = vectorizer.transform(test["predispatch_text"])
    estimator = LogisticRegression(solver="liblinear", class_weight="balanced", C=2.0, max_iter=600, random_state=GLOBAL_SEED)

    service_model = OneVsRestClassifier(estimator, n_jobs=1)
    service_columns = [f"reference_{label}" for label in RESOURCE_LABELS]
    y_train = train[service_columns].to_numpy(int)
    y_validation = validation[service_columns].to_numpy(int)
    y_test = test[service_columns].to_numpy(int)
    service_model.fit(x_train, y_train)
    p_validation = service_model.predict_proba(x_validation)
    p_test = service_model.predict_proba(x_test)
    thresholds = choose_thresholds(y_validation, p_validation)
    pred_validation = p_validation >= np.asarray(thresholds)
    pred_test = p_test >= np.asarray(thresholds)

    fact_model = OneVsRestClassifier(estimator, n_jobs=1)
    fact_columns = [f"fact_{label}" for label in FACT_LABELS]
    fact_model.fit(x_train, train[fact_columns].to_numpy(int))
    p_validation_fact = fact_model.predict_proba(x_validation)
    fact_thresholds = choose_thresholds(validation[fact_columns].to_numpy(int), p_validation_fact)
    pred_test_fact = fact_model.predict_proba(x_test) >= np.asarray(fact_thresholds)

    type_model = LogisticRegression(solver="lbfgs", class_weight="balanced", C=2.0, max_iter=600, random_state=GLOBAL_SEED)
    type_model.fit(x_train, train["incident_type"])
    pred_type = type_model.predict(x_test)

    validation_margin = np.mean(np.maximum(p_validation, 1 - p_validation), axis=1)
    validation_exact = np.all(pred_validation == y_validation, axis=1).astype(int)
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.01, y_max=0.99)
    calibrator.fit(validation_margin, validation_exact)
    test_margin = np.mean(np.maximum(p_test, 1 - p_test), axis=1)
    confidence = calibrator.predict(test_margin)

    exact = np.all(pred_test == y_test, axis=1)
    under = np.any((y_test == 1) & (pred_test == 0), axis=1)
    over = np.any((y_test == 0) & (pred_test == 1), axis=1)
    row_f1 = np.array([f1_score(y_test[i], pred_test[i], zero_division=1) for i in range(len(test))])
    names = vectorizer.get_feature_names_out()
    explanations = [top_lexical_terms(x_test, names, service_model, pred_test, i) for i in range(len(test))]

    test["predicted_incident_type"] = pred_type
    test["predicted_services"] = [[label for label, flag in zip(RESOURCE_LABELS, row) if flag] for row in pred_test]
    test["predicted_material_facts"] = [[label for label, flag in zip(FACT_LABELS, row) if flag] for row in pred_test_fact]
    test["structured_justification"] = explanations
    test["ai_confidence"] = confidence
    test["service_exact_match"] = exact.astype(int)
    test["service_f1"] = row_f1
    test["ai_underdispatch"] = under.astype(int)
    test["ai_overdispatch"] = over.astype(int)
    test["ai_wrong"] = (~exact).astype(int)
    test["category_correct"] = (test["incident_type"].to_numpy() == pred_type).astype(int)
    for index, label in enumerate(RESOURCE_LABELS):
        test[f"prob_{label}"] = p_test[:, index]

    list_columns = [
        "material_facts", "required_information", "missing_information_reference",
        "observed_reference_activation_set", "predicted_services", "predicted_material_facts",
        "structured_justification",
    ]
    benchmark_columns = [
        "seed_incident_id", "source", "date", "year", "title", "predispatch_text",
        "incident_type", "severity", "ambiguous", "protected_predecision", "reversibility",
        *list_columns, "predicted_incident_type", "ai_confidence", "service_exact_match",
        "service_f1", "category_correct", "ai_underdispatch", "ai_overdispatch", "ai_wrong",
        *[f"prob_{label}" for label in RESOURCE_LABELS],
    ]
    benchmark = test[benchmark_columns].copy()
    for column in list_columns:
        benchmark[column] = benchmark[column].map(json.dumps)
    benchmark.to_csv(PREDICTIONS / "frozen_ai_predictions.csv", index=False)
    with (PREDICTIONS / "frozen_ai_predictions.jsonl").open("w", encoding="utf-8") as stream:
        for record in benchmark.to_dict(orient="records"):
            record["date"] = str(record["date"])
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    metrics, bootstrap = bootstrap_ai_metrics(y_test, pred_test.astype(int), confidence, 1000)
    bootstrap.to_csv(STATISTICS / "ai_metric_bootstrap.csv", index=False)
    metrics.update({
        "n_training": int(len(train)),
        "n_validation": int(len(validation)),
        "n_test": int(len(test)),
        "n_test_reserve": int(len(reserve_frame)),
        "underdispatch_rate": float(under.mean()),
        "overdispatch_rate": float(over.mean()),
        "incident_type_accuracy": float(test["category_correct"].mean()),
        "mean_row_f1": float(row_f1.mean()),
        "mean_confidence": float(confidence.mean()),
        "service_thresholds": dict(zip(RESOURCE_LABELS, thresholds)),
        "fact_thresholds": dict(zip(FACT_LABELS, fact_thresholds)),
    })
    (STATISTICS / "revised_ai_evaluation.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    per_service_rows = []
    rng = np.random.default_rng(GLOBAL_SEED + 27)
    for column, label in enumerate(RESOURCE_LABELS):
        precision, recall, f1, _ = precision_recall_fscore_support(y_test[:, column], pred_test[:, column], average="binary", zero_division=0)
        boot_rows = []
        for _ in range(500):
            idx = rng.integers(0, len(test), len(test))
            values = precision_recall_fscore_support(y_test[idx, column], pred_test[idx, column], average="binary", zero_division=0)[:3]
            boot_rows.append(values)
        boot_array = np.asarray(boot_rows)
        per_service_rows.append({
            "service": label,
            "precision": precision,
            "precision_ci_low": np.quantile(boot_array[:, 0], 0.025),
            "precision_ci_high": np.quantile(boot_array[:, 0], 0.975),
            "recall": recall,
            "recall_ci_low": np.quantile(boot_array[:, 1], 0.025),
            "recall_ci_high": np.quantile(boot_array[:, 1], 0.975),
            "f1": f1,
            "f1_ci_low": np.quantile(boot_array[:, 2], 0.025),
            "f1_ci_high": np.quantile(boot_array[:, 2], 0.975),
            "support": int(y_test[:, column].sum()),
            "average_precision": average_precision_score(y_test[:, column], p_test[:, column]),
            "threshold": thresholds[column],
        })
    pd.DataFrame(per_service_rows).to_csv(STATISTICS / "ai_per_service_revised.csv", index=False)

    calibration_rows = []
    exact_int = exact.astype(int)
    edges = np.linspace(0, 1, 11)
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (confidence >= low) & (confidence < high if high < 1 else confidence <= high)
        if mask.any():
            calibration_rows.append({"bin_low": low, "bin_high": high, "n": int(mask.sum()), "mean_confidence": confidence[mask].mean(), "exact_accuracy": exact_int[mask].mean()})
    pd.DataFrame(calibration_rows).to_csv(STATISTICS / "ai_calibration_bins.csv", index=False)

    hyperparameters = {
        "global_seed": GLOBAL_SEED,
        "split": "train <=2022; validation 2023; stratified held-out test n=1200 from 2024-2025 after duplicate/contamination removal",
        "representation": "predispatch narrative only; title excluded; service entities redacted",
        "word_tfidf": {"ngram_range": [1, 2], "min_df": 2, "max_df": 0.98, "max_features": 30000, "sublinear_tf": True},
        "char_tfidf": {"analyzer": "char_wb", "ngram_range": [3, 5], "min_df": 3, "max_features": 25000, "sublinear_tf": True},
        "classifier": {"type": "one-vs-rest logistic regression", "solver": "liblinear", "class_weight": "balanced", "C": 2.0, "max_iter": 600},
        "threshold_selection": "validation-set per-service F1 over 0.10..0.80 in 0.05 steps",
        "confidence": "isotonic calibration of mean service margin to exact-set correctness on 2023 validation set",
    }
    (CONFIGS / "ai_model_hyperparameters.json").write_text(json.dumps(hyperparameters, indent=2), encoding="utf-8")
    return benchmark, metrics


def write_audits(raw_repaired: pd.DataFrame, deduplicated: pd.DataFrame, duplicate_summary: dict) -> dict:
    leakage = {
        "eligible_cases": int(len(raw_repaired)),
        "cases_modified": int(raw_repaired["modified"].sum()),
        "cases_modified_percent": float(raw_repaired["modified"].mean() * 100),
        "cases_with_service_entities_redacted": int(raw_repaired["service_entities_redacted"].gt(0).sum()),
        "service_entities_redacted_total": int(raw_repaired["service_entities_redacted"].sum()),
        "characters_removed_total": int(raw_repaired["characters_removed"].sum()),
        "cases_with_postdecision_preamble_removed": int(raw_repaired["start_offset"].gt(0).sum()),
        "original_downstream_matches": int(raw_repaired["original_downstream_matches"].sum()),
        "remaining_downstream_matches": int(raw_repaired["remaining_downstream_matches"].sum()),
        "remaining_downstream_case_count": int(raw_repaired["remaining_downstream_matches"].gt(0).sum()),
        "title_used_by_model": False,
        "observed_reference_activation_used_as_feature": False,
    }
    (STATISTICS / "leakage_audit_summary.json").write_text(json.dumps(leakage, indent=2), encoding="utf-8")
    audit_columns = [
        "seed_incident_id", "source", "date", "year", "incident_type", "modified", "boundary_rule",
        "start_offset", "characters_removed", "service_entities_redacted", "original_downstream_matches",
        "remaining_downstream_matches", "narrative_original", "predispatch_text",
    ]
    sample_parts = []
    for source, group in raw_repaired.groupby("source"):
        random_part = group.sample(min(10, len(group)), random_state=GLOBAL_SEED).copy()
        random_part["manual_audit_stratum"] = "source_random"
        sample_parts.append(random_part)
        preamble = group.loc[group["start_offset"].gt(0)]
        if len(preamble):
            part = preamble.sample(min(10, len(preamble)), random_state=GLOBAL_SEED + 1).copy()
            part["manual_audit_stratum"] = "postdecision_preamble_removed"
            sample_parts.append(part)
        service = group.loc[group["service_entities_redacted"].gt(0)]
        if len(service):
            part = service.sample(min(10, len(service)), random_state=GLOBAL_SEED + 2).copy()
            part["manual_audit_stratum"] = "service_entity_redacted"
            sample_parts.append(part)
    audit_sample = pd.concat(sample_parts, ignore_index=True).drop_duplicates("seed_incident_id")
    audit_sample = audit_sample[["manual_audit_stratum", *audit_columns]]
    audit_sample.to_csv(STATISTICS / "manual_leakage_audit_sample.csv", index=False)

    weak_rows = []
    for fact in FACT_LABELS:
        weak_rows.append({
            "label": fact,
            "positive_original_old_rule": int(raw_repaired[f"fact_old_original_{fact}"].sum()),
            "positive_predispatch_old_rule": int(raw_repaired[f"fact_old_predispatch_{fact}"].sum()),
            "positive_predispatch_corrected_rule": int(raw_repaired[f"fact_{fact}"].sum()),
            "changed_due_predispatch_redaction": int((raw_repaired[f"fact_old_original_{fact}"] != raw_repaired[f"fact_old_predispatch_{fact}"]).sum()),
            "changed_due_rule_correction": int((raw_repaired[f"fact_old_predispatch_{fact}"] != raw_repaired[f"fact_{fact}"]).sum()),
        })
    weak = pd.DataFrame(weak_rows)
    weak.to_csv(STATISTICS / "weak_label_audit.csv", index=False)
    corrected_mask = raw_repaired["fact_old_predispatch_water_drowning"] != raw_repaired["fact_water_drowning"]
    raw_repaired.loc[corrected_mask, ["seed_incident_id", "date", "predispatch_text", "fact_old_predispatch_water_drowning", "fact_water_drowning"]].sample(
        min(40, int(corrected_mask.sum())), random_state=GLOBAL_SEED
    ).to_csv(STATISTICS / "manual_weak_label_corrected_sample.csv", index=False)

    severity_counts = raw_repaired["severity"].value_counts().rename_axis("severity").reset_index(name="n")
    severity_counts["share"] = severity_counts["n"] / severity_counts["n"].sum()
    severity_counts.to_csv(STATISTICS / "predecision_severity_distribution.csv", index=False)
    severity_config = {
        "input": "predispatch_text only",
        "target_activation_features_used": False,
        "ai_correctness_used": False,
        "downstream_intervention_used": False,
        "rules": {"S2": S2_RE.pattern, "S0": S0_RE.pattern, "default": "S1"},
    }
    (CONFIGS / "predecision_severity_rules.json").write_text(json.dumps(severity_config, indent=2), encoding="utf-8")
    (STATISTICS / "duplicate_audit_summary.json").write_text(json.dumps(duplicate_summary, indent=2), encoding="utf-8")
    return {"leakage": leakage, "duplicates": duplicate_summary, "weak_labels": weak_rows}


def write_frozen_manifest(benchmark: pd.DataFrame, metrics: dict, audits: dict) -> None:
    input_files = sorted(RAW.glob("*"))
    output_files = [
        PROCESSED / "repaired_incident_dataset.csv",
        PROCESSED / "frozen_split_assignments.csv",
        PREDICTIONS / "frozen_ai_predictions.csv",
        STATISTICS / "revised_ai_evaluation.json",
    ]
    manifest = {
        "system": "OISE revised capacity-aware authority experiment",
        "execution_date": EXECUTION_DATE,
        "global_seed": GLOBAL_SEED,
        "python": sys.version,
        "platform": platform.platform(),
        "versions": {"numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__, "scikit_learn": sklearn.__version__},
        "input_hashes": {str(path.relative_to(ROOT)): sha256(path) for path in input_files},
        "frozen_output_hashes": {str(path.relative_to(ROOT)): sha256(path) for path in output_files},
        "benchmark_n": int(len(benchmark)),
        "ai_metrics": metrics,
        "audit_summary": audits,
        "regime_invariance_rule": "Every authority regime must reuse this exact frozen prediction file and global seed schedule.",
    }
    (CONFIGS / "frozen_system_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    repaired = load_and_repair()
    deduplicated, _, duplicate_summary = duplicate_audit(repaired)
    audits = write_audits(repaired, deduplicated, duplicate_summary)
    if audits["leakage"]["remaining_downstream_matches"] != 0:
        raise RuntimeError(
            "Leakage audit failed: "
            f"{audits['leakage']['remaining_downstream_matches']} downstream markers remain"
        )
    serializable = deduplicated.copy()
    for column in ["observed_reference_activation_set", "material_facts", "required_information", "missing_information_reference"]:
        serializable[column] = serializable[column].map(json.dumps)
    serializable.to_csv(PROCESSED / "repaired_incident_dataset.csv", index=False)
    benchmark, metrics = train_and_freeze(deduplicated)
    preaudit = json.loads((CONFIGS / "preaudit_ai_evaluation.json").read_text(encoding="utf-8"))
    comparison_rows = []
    mapping = {
        "exact_activation_set_accuracy": "service_exact_match",
        "micro_f1": "service_micro_f1",
        "macro_f1": "service_macro_f1",
        "confidence_brier": "confidence_brier",
        "confidence_ece_10": "confidence_ece_10",
    }
    for revised_name, old_name in mapping.items():
        revised_value = metrics[revised_name]["estimate"]
        comparison_rows.append({
            "metric": revised_name,
            "preaudit_contaminated_estimate": preaudit.get(old_name),
            "revised_leakage_free_estimate": revised_value,
            "absolute_change": revised_value - preaudit.get(old_name, np.nan),
            "preaudit_valid_for_substantive_use": False,
        })
    pd.DataFrame(comparison_rows).to_csv(STATISTICS / "preaudit_vs_revised_ai.csv", index=False)
    write_frozen_manifest(benchmark, metrics, audits)
    print(json.dumps({
        "status": "PASS",
        "eligible_before_duplicate_audit": len(repaired),
        "eligible_after_duplicate_audit": len(deduplicated),
        "train": metrics["n_training"],
        "validation": metrics["n_validation"],
        "test": metrics["n_test"],
        "exact_accuracy": metrics["exact_activation_set_accuracy"]["estimate"],
        "micro_f1": metrics["micro_f1"]["estimate"],
        "leakage_cases_modified_percent": audits["leakage"]["cases_modified_percent"],
    }, indent=2))


if __name__ == "__main__":
    main()
