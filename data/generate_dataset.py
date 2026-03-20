import pandas as pd
import numpy as np
import os
import random
import math
from typing import Dict, Any, List

# ---------------------------------------------------------------------------
# FERTILIZER TYPES
# ---------------------------------------------------------------------------
FERTILIZER_TYPES = [
    "Urea", "DAP", "MOP", "SSP", "TSP", "10-26-26", "14-35-14", "19-19-19",
    "20-20-20", "Ammonium_Sulfate", "Calcium_Nitrate", "Potassium_Sulfate",
    "Compost", "Vermicompost", "Neem_Cake"
]

ORGANIC_CROPS = {
    "sweet_basil", "thai_basil", "oregano", "thyme", "rosemary", "sage",
    "marjoram", "flat_parsley", "curly_parsley", "cilantro", "dill",
    "chervil", "tarragon", "lavender", "lemon_balm", "peppermint",
    "spearmint", "fennel_herb", "borage", "chamomile",
    "aloe_vera", "echinacea", "american_ginseng", "ashwagandha", "moringa",
    "st_johns_wort", "valerian", "milk_thistle", "feverfew", "goldenseal",
    "astragalus", "rhodiola", "stinging_nettle", "skullcap", "witch_hazel",
    "ginger", "turmeric", "cardamom", "vanilla", "cinnamon", "clove",
    "coriander_seed", "cumin", "fenugreek", "star_anise", "lemongrass",
    "galangal", "wasabi", "saffron", "allspice"
}

SOIL_TYPES = [
    "clay", "sandy", "loamy", "black", "red", "alluvial",
    "desert", "peat", "chalk", "silt", "laterite", "saline", "acidic", "alkaline"
]

SEASONS = ["Kharif", "Rabi", "Zaid", "Perennial", "Spring", "Summer", "Autumn", "Winter"]

REGIONS = [
    "tropical", "temperate", "arid", "semi_arid", "subtropical",
    "mediterranean", "humid_subtropical", "highland", "coastal",
    "continental", "boreal", "steppe"
]

# ---------------------------------------------------------------------------
# BASE_SPECIES  keys: N_range, P_range, K_range (kg/ha), temp (°C mean),
#   humidity (%), ph (mean), rain (mm/month), Kc, growing_days, fert, latitude
# ---------------------------------------------------------------------------
BASE_SPECIES = {
    # ===== CEREALS (25) =====
    "rice":             {"N_range":(80,120),  "P_range":(35,60),   "K_range":(35,50),   "temp":24, "humidity":82, "ph":6.2, "rain":250, "Kc":1.20, "growing_days":120, "fert":"Urea",          "latitude":15},
    "wheat_bread":      {"N_range":(60,100),  "P_range":(30,55),   "K_range":(30,55),   "temp":18, "humidity":60, "ph":6.5, "rain":75,  "Kc":1.10, "growing_days":130, "fert":"DAP",           "latitude":40},
    "wheat_durum":      {"N_range":(70,110),  "P_range":(30,55),   "K_range":(30,55),   "temp":20, "humidity":55, "ph":6.8, "rain":65,  "Kc":1.10, "growing_days":125, "fert":"DAP",           "latitude":38},
    "wheat_spelt":      {"N_range":(50,90),   "P_range":(25,50),   "K_range":(25,50),   "temp":16, "humidity":58, "ph":6.5, "rain":70,  "Kc":1.05, "growing_days":135, "fert":"DAP",           "latitude":45},
    "barley_spring":    {"N_range":(60,100),  "P_range":(25,50),   "K_range":(30,55),   "temp":15, "humidity":55, "ph":6.5, "rain":60,  "Kc":1.00, "growing_days":90,  "fert":"Urea",          "latitude":50},
    "barley_winter":    {"N_range":(70,110),  "P_range":(25,50),   "K_range":(30,55),   "temp":12, "humidity":60, "ph":6.5, "rain":65,  "Kc":1.00, "growing_days":200, "fert":"Urea",          "latitude":48},
    "oats":             {"N_range":(50,90),   "P_range":(20,45),   "K_range":(25,50),   "temp":14, "humidity":65, "ph":6.2, "rain":70,  "Kc":1.05, "growing_days":100, "fert":"Urea",          "latitude":52},
    "rye":              {"N_range":(50,90),   "P_range":(20,45),   "K_range":(30,55),   "temp":12, "humidity":60, "ph":5.8, "rain":65,  "Kc":1.00, "growing_days":120, "fert":"Urea",          "latitude":55},
    "corn_sweet":       {"N_range":(100,150), "P_range":(40,65),   "K_range":(40,65),   "temp":24, "humidity":65, "ph":6.2, "rain":90,  "Kc":1.15, "growing_days":80,  "fert":"10-26-26",      "latitude":35},
    "corn_field":       {"N_range":(120,180), "P_range":(50,75),   "K_range":(50,80),   "temp":25, "humidity":68, "ph":6.2, "rain":100, "Kc":1.20, "growing_days":120, "fert":"10-26-26",      "latitude":38},
    "sorghum_grain":    {"N_range":(80,120),  "P_range":(30,55),   "K_range":(30,55),   "temp":28, "humidity":55, "ph":6.5, "rain":60,  "Kc":1.00, "growing_days":110, "fert":"Urea",          "latitude":20},
    "sorghum_sweet":    {"N_range":(90,130),  "P_range":(35,60),   "K_range":(35,60),   "temp":30, "humidity":58, "ph":6.5, "rain":70,  "Kc":1.10, "growing_days":120, "fert":"Urea",          "latitude":18},
    "pearl_millet":     {"N_range":(60,100),  "P_range":(25,45),   "K_range":(25,45),   "temp":30, "humidity":50, "ph":6.5, "rain":45,  "Kc":0.95, "growing_days":90,  "fert":"Urea",          "latitude":15},
    "finger_millet":    {"N_range":(40,80),   "P_range":(20,40),   "K_range":(20,40),   "temp":27, "humidity":60, "ph":6.0, "rain":60,  "Kc":0.90, "growing_days":100, "fert":"Urea",          "latitude":12},
    "foxtail_millet":   {"N_range":(40,80),   "P_range":(20,40),   "K_range":(20,40),   "temp":26, "humidity":55, "ph":6.2, "rain":50,  "Kc":0.90, "growing_days":90,  "fert":"Urea",          "latitude":25},
    "proso_millet":     {"N_range":(35,70),   "P_range":(15,35),   "K_range":(15,35),   "temp":24, "humidity":50, "ph":6.2, "rain":40,  "Kc":0.85, "growing_days":80,  "fert":"Urea",          "latitude":30},
    "teff":             {"N_range":(40,80),   "P_range":(15,35),   "K_range":(15,35),   "temp":20, "humidity":55, "ph":6.0, "rain":55,  "Kc":0.90, "growing_days":90,  "fert":"Urea",          "latitude":10},
    "quinoa":           {"N_range":(50,90),   "P_range":(20,45),   "K_range":(30,55),   "temp":15, "humidity":50, "ph":6.5, "rain":50,  "Kc":0.95, "growing_days":120, "fert":"Compost",       "latitude":-15},
    "amaranth":         {"N_range":(60,100),  "P_range":(25,50),   "K_range":(25,50),   "temp":25, "humidity":55, "ph":6.5, "rain":55,  "Kc":0.95, "growing_days":100, "fert":"Urea",          "latitude":20},
    "buckwheat":        {"N_range":(30,70),   "P_range":(15,35),   "K_range":(25,50),   "temp":18, "humidity":60, "ph":6.0, "rain":65,  "Kc":0.90, "growing_days":70,  "fert":"Compost",       "latitude":45},
    "triticale":        {"N_range":(60,100),  "P_range":(25,50),   "K_range":(30,55),   "temp":16, "humidity":60, "ph":6.2, "rain":70,  "Kc":1.05, "growing_days":130, "fert":"DAP",           "latitude":47},
    "barnyard_millet":  {"N_range":(40,75),   "P_range":(15,35),   "K_range":(15,35),   "temp":28, "humidity":60, "ph":6.0, "rain":55,  "Kc":0.90, "growing_days":85,  "fert":"Urea",          "latitude":15},
    "kodo_millet":      {"N_range":(35,70),   "P_range":(15,35),   "K_range":(15,35),   "temp":27, "humidity":60, "ph":5.8, "rain":50,  "Kc":0.88, "growing_days":100, "fert":"Urea",          "latitude":18},
    "fonio":            {"N_range":(25,55),   "P_range":(10,30),   "K_range":(10,30),   "temp":28, "humidity":65, "ph":5.5, "rain":60,  "Kc":0.85, "growing_days":70,  "fert":"Compost",       "latitude":12},
    "spelt":            {"N_range":(50,90),   "P_range":(25,50),   "K_range":(25,50),   "temp":16, "humidity":58, "ph":6.3, "rain":70,  "Kc":1.05, "growing_days":135, "fert":"DAP",           "latitude":46},

    # ===== LEGUMES (30) =====
    "soybean":          {"N_range":(20,50),   "P_range":(35,65),   "K_range":(40,70),   "temp":24, "humidity":65, "ph":6.5, "rain":100, "Kc":1.10, "growing_days":120, "fert":"SSP",           "latitude":30},
    "peanut":           {"N_range":(15,40),   "P_range":(30,55),   "K_range":(35,65),   "temp":28, "humidity":60, "ph":6.2, "rain":70,  "Kc":1.05, "growing_days":130, "fert":"SSP",           "latitude":20},
    "common_bean":      {"N_range":(20,45),   "P_range":(40,65),   "K_range":(30,55),   "temp":20, "humidity":60, "ph":6.2, "rain":80,  "Kc":1.00, "growing_days":90,  "fert":"DAP",           "latitude":25},
    "kidney_bean":      {"N_range":(20,45),   "P_range":(60,80),   "K_range":(15,30),   "temp":20, "humidity":60, "ph":6.0, "rain":80,  "Kc":1.00, "growing_days":100, "fert":"Compost",       "latitude":30},
    "navy_bean":        {"N_range":(20,45),   "P_range":(45,70),   "K_range":(25,50),   "temp":19, "humidity":60, "ph":6.2, "rain":75,  "Kc":0.98, "growing_days":95,  "fert":"DAP",           "latitude":40},
    "black_bean":       {"N_range":(20,45),   "P_range":(40,65),   "K_range":(25,50),   "temp":22, "humidity":65, "ph":6.2, "rain":85,  "Kc":1.00, "growing_days":100, "fert":"DAP",           "latitude":20},
    "lima_bean":        {"N_range":(20,45),   "P_range":(40,65),   "K_range":(30,55),   "temp":24, "humidity":65, "ph":6.0, "rain":80,  "Kc":1.00, "growing_days":110, "fert":"SSP",           "latitude":15},
    "fava_bean":        {"N_range":(25,55),   "P_range":(35,60),   "K_range":(30,55),   "temp":16, "humidity":65, "ph":6.5, "rain":70,  "Kc":1.05, "growing_days":120, "fert":"DAP",           "latitude":40},
    "lentil_red":       {"N_range":(15,35),   "P_range":(45,70),   "K_range":(20,45),   "temp":20, "humidity":60, "ph":6.5, "rain":50,  "Kc":0.95, "growing_days":110, "fert":"SSP",           "latitude":35},
    "lentil_green":     {"N_range":(15,35),   "P_range":(50,75),   "K_range":(20,45),   "temp":18, "humidity":58, "ph":6.3, "rain":55,  "Kc":0.95, "growing_days":115, "fert":"SSP",           "latitude":38},
    "chickpea":         {"N_range":(20,50),   "P_range":(35,60),   "K_range":(25,45),   "temp":24, "humidity":50, "ph":6.5, "rain":70,  "Kc":1.00, "growing_days":100, "fert":"SSP",           "latitude":28},
    "green_pea":        {"N_range":(20,45),   "P_range":(35,60),   "K_range":(30,55),   "temp":15, "humidity":65, "ph":6.2, "rain":65,  "Kc":1.05, "growing_days":80,  "fert":"DAP",           "latitude":45},
    "snap_pea":         {"N_range":(20,45),   "P_range":(35,60),   "K_range":(30,55),   "temp":15, "humidity":65, "ph":6.2, "rain":65,  "Kc":1.00, "growing_days":75,  "fert":"DAP",           "latitude":45},
    "mung_bean":        {"N_range":(15,35),   "P_range":(40,60),   "K_range":(15,30),   "temp":28, "humidity":85, "ph":7.0, "rain":50,  "Kc":0.95, "growing_days":70,  "fert":"DAP",           "latitude":20},
    "adzuki_bean":      {"N_range":(20,45),   "P_range":(40,65),   "K_range":(25,50),   "temp":22, "humidity":65, "ph":6.5, "rain":80,  "Kc":1.00, "growing_days":120, "fert":"DAP",           "latitude":35},
    "cowpea":           {"N_range":(15,40),   "P_range":(30,55),   "K_range":(20,45),   "temp":28, "humidity":55, "ph":6.5, "rain":50,  "Kc":0.95, "growing_days":90,  "fert":"SSP",           "latitude":12},
    "pigeon_pea":       {"N_range":(20,45),   "P_range":(60,80),   "K_range":(15,30),   "temp":25, "humidity":40, "ph":6.0, "rain":120, "Kc":1.00, "growing_days":180, "fert":"SSP",           "latitude":15},
    "moth_bean":        {"N_range":(20,45),   "P_range":(35,60),   "K_range":(15,30),   "temp":27, "humidity":50, "ph":7.0, "rain":50,  "Kc":0.90, "growing_days":90,  "fert":"MOP",           "latitude":22},
    "tepary_bean":      {"N_range":(15,40),   "P_range":(30,55),   "K_range":(20,45),   "temp":30, "humidity":40, "ph":7.0, "rain":30,  "Kc":0.85, "growing_days":80,  "fert":"SSP",           "latitude":28},
    "jack_bean":        {"N_range":(20,45),   "P_range":(30,55),   "K_range":(20,45),   "temp":26, "humidity":70, "ph":6.0, "rain":100, "Kc":1.00, "growing_days":180, "fert":"Compost",       "latitude":10},
    "velvet_bean":      {"N_range":(20,50),   "P_range":(30,55),   "K_range":(20,45),   "temp":26, "humidity":70, "ph":6.0, "rain":100, "Kc":1.05, "growing_days":150, "fert":"Compost",       "latitude":12},
    "hyacinth_bean":    {"N_range":(20,45),   "P_range":(30,55),   "K_range":(20,45),   "temp":28, "humidity":65, "ph":6.2, "rain":80,  "Kc":0.95, "growing_days":120, "fert":"DAP",           "latitude":15},
    "winged_bean":      {"N_range":(20,50),   "P_range":(35,60),   "K_range":(25,50),   "temp":27, "humidity":80, "ph":6.0, "rain":100, "Kc":1.05, "growing_days":120, "fert":"SSP",           "latitude":10},
    "bambara_groundnut":{"N_range":(15,40),   "P_range":(25,50),   "K_range":(20,45),   "temp":28, "humidity":60, "ph":6.5, "rain":60,  "Kc":0.90, "growing_days":150, "fert":"SSP",           "latitude":12},
    "grass_pea":        {"N_range":(15,35),   "P_range":(30,55),   "K_range":(20,45),   "temp":18, "humidity":55, "ph":6.2, "rain":55,  "Kc":0.90, "growing_days":120, "fert":"DAP",           "latitude":35},
    "horse_gram":       {"N_range":(15,35),   "P_range":(25,50),   "K_range":(15,35),   "temp":28, "humidity":55, "ph":6.5, "rain":50,  "Kc":0.88, "growing_days":120, "fert":"SSP",           "latitude":18},
    "cluster_bean":     {"N_range":(15,35),   "P_range":(25,50),   "K_range":(20,40),   "temp":30, "humidity":45, "ph":7.0, "rain":40,  "Kc":0.88, "growing_days":90,  "fert":"SSP",           "latitude":25},
    "lablab_bean":      {"N_range":(20,45),   "P_range":(30,55),   "K_range":(20,45),   "temp":25, "humidity":65, "ph":6.0, "rain":80,  "Kc":0.95, "growing_days":120, "fert":"DAP",           "latitude":12},
    "sword_bean":       {"N_range":(20,45),   "P_range":(30,55),   "K_range":(20,45),   "temp":26, "humidity":70, "ph":6.0, "rain":90,  "Kc":1.00, "growing_days":150, "fert":"Compost",       "latitude":10},
    "scarlet_runner_bean":{"N_range":(20,45), "P_range":(35,60),   "K_range":(25,50),   "temp":18, "humidity":65, "ph":6.2, "rain":75,  "Kc":1.00, "growing_days":100, "fert":"DAP",           "latitude":42},

    # ===== SOLANACEAE (20) =====
    "tomato":           {"N_range":(100,150), "P_range":(40,60),   "K_range":(180,220), "temp":24, "humidity":70, "ph":6.4, "rain":50,  "Kc":1.15, "growing_days":90,  "fert":"MOP",           "latitude":35},
    "cherry_tomato":    {"N_range":(90,130),  "P_range":(40,60),   "K_range":(160,200), "temp":24, "humidity":70, "ph":6.3, "rain":50,  "Kc":1.10, "growing_days":80,  "fert":"MOP",           "latitude":35},
    "beef_tomato":      {"N_range":(110,160), "P_range":(45,65),   "K_range":(190,230), "temp":25, "humidity":70, "ph":6.4, "rain":55,  "Kc":1.20, "growing_days":100, "fert":"MOP",           "latitude":35},
    "roma_tomato":      {"N_range":(100,145), "P_range":(40,60),   "K_range":(175,215), "temp":25, "humidity":68, "ph":6.4, "rain":50,  "Kc":1.15, "growing_days":85,  "fert":"MOP",           "latitude":35},
    "bell_pepper":      {"N_range":(120,160), "P_range":(60,80),   "K_range":(160,200), "temp":25, "humidity":70, "ph":6.4, "rain":65,  "Kc":1.10, "growing_days":100, "fert":"10-26-26",      "latitude":32},
    "jalapeno":         {"N_range":(100,140), "P_range":(50,70),   "K_range":(140,180), "temp":26, "humidity":65, "ph":6.5, "rain":60,  "Kc":1.05, "growing_days":90,  "fert":"10-26-26",      "latitude":20},
    "habanero":         {"N_range":(90,130),  "P_range":(45,65),   "K_range":(130,170), "temp":28, "humidity":65, "ph":6.5, "rain":55,  "Kc":1.05, "growing_days":100, "fert":"10-26-26",      "latitude":18},
    "cayenne":          {"N_range":(100,140), "P_range":(45,65),   "K_range":(135,175), "temp":27, "humidity":65, "ph":6.5, "rain":60,  "Kc":1.05, "growing_days":95,  "fert":"10-26-26",      "latitude":20},
    "serrano":          {"N_range":(90,130),  "P_range":(45,65),   "K_range":(130,170), "temp":26, "humidity":65, "ph":6.5, "rain":55,  "Kc":1.00, "growing_days":90,  "fert":"10-26-26",      "latitude":22},
    "eggplant":         {"N_range":(120,160), "P_range":(50,70),   "K_range":(160,200), "temp":26, "humidity":65, "ph":6.0, "rain":60,  "Kc":1.10, "growing_days":100, "fert":"14-35-14",      "latitude":28},
    "japanese_eggplant":{"N_range":(110,150), "P_range":(50,70),   "K_range":(150,190), "temp":26, "humidity":65, "ph":6.0, "rain":60,  "Kc":1.10, "growing_days":95,  "fert":"14-35-14",      "latitude":35},
    "thai_eggplant":    {"N_range":(100,140), "P_range":(45,65),   "K_range":(140,180), "temp":28, "humidity":70, "ph":6.0, "rain":65,  "Kc":1.05, "growing_days":90,  "fert":"14-35-14",      "latitude":15},
    "potato_russet":    {"N_range":(120,160), "P_range":(80,100),  "K_range":(180,220), "temp":17, "humidity":60, "ph":5.8, "rain":65,  "Kc":1.15, "growing_days":100, "fert":"10-26-26",      "latitude":40},
    "potato_yukon":     {"N_range":(120,160), "P_range":(80,100),  "K_range":(180,220), "temp":17, "humidity":60, "ph":6.0, "rain":65,  "Kc":1.15, "growing_days":90,  "fert":"10-26-26",      "latitude":40},
    "sweet_potato":     {"N_range":(60,100),  "P_range":(30,55),   "K_range":(100,150), "temp":26, "humidity":70, "ph":6.0, "rain":70,  "Kc":1.05, "growing_days":120, "fert":"MOP",           "latitude":20},
    "physalis":         {"N_range":(80,120),  "P_range":(35,60),   "K_range":(100,140), "temp":22, "humidity":60, "ph":6.5, "rain":60,  "Kc":1.00, "growing_days":100, "fert":"Compost",       "latitude":30},
    "tomatillo":        {"N_range":(100,140), "P_range":(40,65),   "K_range":(120,160), "temp":24, "humidity":65, "ph":6.5, "rain":60,  "Kc":1.05, "growing_days":90,  "fert":"10-26-26",      "latitude":22},
    "cape_gooseberry":  {"N_range":(80,120),  "P_range":(35,60),   "K_range":(100,140), "temp":20, "humidity":60, "ph":6.2, "rain":65,  "Kc":1.00, "growing_days":120, "fert":"Compost",       "latitude":-30},
    "ghost_pepper":     {"N_range":(90,130),  "P_range":(45,65),   "K_range":(130,170), "temp":27, "humidity":70, "ph":6.5, "rain":65,  "Kc":1.05, "growing_days":120, "fert":"10-26-26",      "latitude":26},
    "banana_pepper":    {"N_range":(90,130),  "P_range":(45,65),   "K_range":(130,170), "temp":22, "humidity":65, "ph":6.5, "rain":60,  "Kc":1.00, "growing_days":85,  "fert":"10-26-26",      "latitude":38},

    # ===== ROOT VEGETABLES (20) =====
    "carrot":           {"N_range":(80,120),  "P_range":(40,60),   "K_range":(120,160), "temp":18, "humidity":70, "ph":6.2, "rain":50,  "Kc":1.05, "growing_days":90,  "fert":"MOP",           "latitude":42},
    "beet":             {"N_range":(100,140), "P_range":(35,60),   "K_range":(130,170), "temp":18, "humidity":65, "ph":6.5, "rain":55,  "Kc":1.05, "growing_days":90,  "fert":"Urea",          "latitude":45},
    "radish":           {"N_range":(60,100),  "P_range":(30,55),   "K_range":(80,120),  "temp":16, "humidity":65, "ph":6.2, "rain":40,  "Kc":0.95, "growing_days":30,  "fert":"Urea",          "latitude":40},
    "daikon":           {"N_range":(70,110),  "P_range":(30,55),   "K_range":(90,130),  "temp":16, "humidity":65, "ph":6.2, "rain":45,  "Kc":0.95, "growing_days":60,  "fert":"Urea",          "latitude":35},
    "turnip":           {"N_range":(60,100),  "P_range":(30,55),   "K_range":(80,120),  "temp":15, "humidity":65, "ph":6.5, "rain":50,  "Kc":0.95, "growing_days":60,  "fert":"Urea",          "latitude":45},
    "rutabaga":         {"N_range":(70,110),  "P_range":(30,55),   "K_range":(90,130),  "temp":14, "humidity":65, "ph":6.5, "rain":55,  "Kc":1.00, "growing_days":90,  "fert":"Urea",          "latitude":50},
    "parsnip":          {"N_range":(70,110),  "P_range":(35,60),   "K_range":(100,140), "temp":16, "humidity":65, "ph":6.5, "rain":55,  "Kc":1.00, "growing_days":120, "fert":"MOP",           "latitude":50},
    "celeriac":         {"N_range":(100,140), "P_range":(40,65),   "K_range":(120,160), "temp":16, "humidity":70, "ph":6.5, "rain":65,  "Kc":1.05, "growing_days":120, "fert":"Urea",          "latitude":48},
    "fennel_bulb":      {"N_range":(60,100),  "P_range":(30,55),   "K_range":(80,120),  "temp":18, "humidity":60, "ph":6.5, "rain":50,  "Kc":1.00, "growing_days":90,  "fert":"Compost",       "latitude":40},
    "jicama":           {"N_range":(50,90),   "P_range":(25,50),   "K_range":(70,110),  "temp":26, "humidity":70, "ph":6.5, "rain":80,  "Kc":1.00, "growing_days":180, "fert":"SSP",           "latitude":20},
    "cassava":          {"N_range":(60,100),  "P_range":(25,50),   "K_range":(80,130),  "temp":27, "humidity":70, "ph":6.0, "rain":100, "Kc":1.10, "growing_days":300, "fert":"MOP",           "latitude":10},
    "yam":              {"N_range":(70,110),  "P_range":(30,55),   "K_range":(100,150), "temp":28, "humidity":75, "ph":5.5, "rain":120, "Kc":1.10, "growing_days":240, "fert":"MOP",           "latitude":8},
    "taro":             {"N_range":(80,120),  "P_range":(35,60),   "K_range":(100,150), "temp":27, "humidity":80, "ph":5.5, "rain":150, "Kc":1.15, "growing_days":240, "fert":"MOP",           "latitude":10},
    "jerusalem_artichoke":{"N_range":(70,110),"P_range":(30,55),   "K_range":(100,150), "temp":18, "humidity":60, "ph":6.5, "rain":60,  "Kc":1.00, "growing_days":150, "fert":"Compost",       "latitude":40},
    "horseradish":      {"N_range":(60,100),  "P_range":(30,55),   "K_range":(80,120),  "temp":16, "humidity":65, "ph":6.0, "rain":60,  "Kc":1.00, "growing_days":150, "fert":"Compost",       "latitude":45},
    "skirret":          {"N_range":(50,90),   "P_range":(25,50),   "K_range":(70,110),  "temp":16, "humidity":65, "ph":6.5, "rain":55,  "Kc":0.95, "growing_days":150, "fert":"Compost",       "latitude":48},
    "scorzonera":       {"N_range":(50,90),   "P_range":(25,50),   "K_range":(70,110),  "temp":16, "humidity":60, "ph":6.2, "rain":50,  "Kc":0.95, "growing_days":180, "fert":"Compost",       "latitude":45},
    "salsify":          {"N_range":(50,90),   "P_range":(25,50),   "K_range":(70,110),  "temp":16, "humidity":60, "ph":6.5, "rain":50,  "Kc":0.95, "growing_days":150, "fert":"Compost",       "latitude":45},
    "water_chestnut":   {"N_range":(60,100),  "P_range":(30,55),   "K_range":(80,120),  "temp":25, "humidity":85, "ph":6.5, "rain":200, "Kc":1.15, "growing_days":150, "fert":"Urea",          "latitude":25},
    "arrowroot":        {"N_range":(50,90),   "P_range":(25,50),   "K_range":(80,130),  "temp":26, "humidity":80, "ph":6.0, "rain":130, "Kc":1.05, "growing_days":180, "fert":"Compost",       "latitude":15},

    # ===== BRASSICAS (20) =====
    "cabbage":          {"N_range":(180,220), "P_range":(60,80),   "K_range":(180,220), "temp":18, "humidity":70, "ph":6.5, "rain":70,  "Kc":1.05, "growing_days":90,  "fert":"Urea",          "latitude":45},
    "red_cabbage":      {"N_range":(170,210), "P_range":(60,80),   "K_range":(170,210), "temp":17, "humidity":70, "ph":6.5, "rain":70,  "Kc":1.05, "growing_days":95,  "fert":"Urea",          "latitude":45},
    "savoy_cabbage":    {"N_range":(175,215), "P_range":(60,80),   "K_range":(175,215), "temp":16, "humidity":70, "ph":6.5, "rain":70,  "Kc":1.05, "growing_days":90,  "fert":"Urea",          "latitude":48},
    "napa_cabbage":     {"N_range":(160,200), "P_range":(55,75),   "K_range":(160,200), "temp":17, "humidity":70, "ph":6.5, "rain":65,  "Kc":1.00, "growing_days":80,  "fert":"Urea",          "latitude":38},
    "kale":             {"N_range":(150,200), "P_range":(55,75),   "K_range":(150,200), "temp":15, "humidity":70, "ph":6.5, "rain":75,  "Kc":1.00, "growing_days":80,  "fert":"Urea",          "latitude":50},
    "broccoli":         {"N_range":(160,200), "P_range":(60,80),   "K_range":(160,200), "temp":18, "humidity":70, "ph":6.4, "rain":70,  "Kc":1.05, "growing_days":80,  "fert":"10-26-26",      "latitude":45},
    "cauliflower":      {"N_range":(170,210), "P_range":(60,80),   "K_range":(170,210), "temp":17, "humidity":70, "ph":6.5, "rain":70,  "Kc":1.05, "growing_days":80,  "fert":"10-26-26",      "latitude":45},
    "brussels_sprouts": {"N_range":(180,220), "P_range":(60,80),   "K_range":(180,220), "temp":15, "humidity":70, "ph":6.5, "rain":75,  "Kc":1.05, "growing_days":120, "fert":"Urea",          "latitude":50},
    "mustard_greens":   {"N_range":(120,160), "P_range":(40,65),   "K_range":(100,140), "temp":18, "humidity":65, "ph":6.2, "rain":60,  "Kc":1.00, "growing_days":60,  "fert":"Urea",          "latitude":35},
    "collard_greens":   {"N_range":(130,170), "P_range":(45,70),   "K_range":(110,150), "temp":20, "humidity":65, "ph":6.5, "rain":65,  "Kc":1.00, "growing_days":75,  "fert":"Urea",          "latitude":35},
    "pak_choi":         {"N_range":(100,140), "P_range":(40,65),   "K_range":(90,130),  "temp":18, "humidity":70, "ph":6.5, "rain":60,  "Kc":0.95, "growing_days":50,  "fert":"Urea",          "latitude":30},
    "choy_sum":         {"N_range":(100,140), "P_range":(40,65),   "K_range":(90,130),  "temp":20, "humidity":70, "ph":6.5, "rain":60,  "Kc":0.95, "growing_days":50,  "fert":"Urea",          "latitude":25},
    "tatsoi":           {"N_range":(90,130),  "P_range":(35,60),   "K_range":(80,120),  "temp":16, "humidity":70, "ph":6.5, "rain":55,  "Kc":0.95, "growing_days":45,  "fert":"Compost",       "latitude":35},
    "mizuna":           {"N_range":(90,130),  "P_range":(35,60),   "K_range":(80,120),  "temp":16, "humidity":70, "ph":6.5, "rain":55,  "Kc":0.95, "growing_days":40,  "fert":"Compost",       "latitude":35},
    "komatsuna":        {"N_range":(90,130),  "P_range":(35,60),   "K_range":(80,120),  "temp":18, "humidity":70, "ph":6.5, "rain":55,  "Kc":0.95, "growing_days":40,  "fert":"Compost",       "latitude":35},
    "kohlrabi":         {"N_range":(130,170), "P_range":(50,70),   "K_range":(120,160), "temp":17, "humidity":65, "ph":6.5, "rain":65,  "Kc":1.00, "growing_days":60,  "fert":"Urea",          "latitude":45},
    "romanesco":        {"N_range":(160,200), "P_range":(60,80),   "K_range":(155,195), "temp":17, "humidity":70, "ph":6.5, "rain":70,  "Kc":1.05, "growing_days":90,  "fert":"10-26-26",      "latitude":45},
    "purple_broccoli":  {"N_range":(155,195), "P_range":(60,80),   "K_range":(155,195), "temp":17, "humidity":70, "ph":6.4, "rain":70,  "Kc":1.05, "growing_days":85,  "fert":"10-26-26",      "latitude":45},
    "garden_cress":     {"N_range":(80,120),  "P_range":(30,55),   "K_range":(70,110),  "temp":16, "humidity":65, "ph":6.2, "rain":50,  "Kc":0.90, "growing_days":25,  "fert":"Compost",       "latitude":45},
    "land_cress":       {"N_range":(80,120),  "P_range":(30,55),   "K_range":(70,110),  "temp":15, "humidity":70, "ph":6.2, "rain":55,  "Kc":0.90, "growing_days":30,  "fert":"Compost",       "latitude":48},

    # ===== ALLIUMS (10) =====
    "onion":            {"N_range":(100,120), "P_range":(40,60),   "K_range":(100,140), "temp":20, "humidity":60, "ph":6.5, "rain":40,  "Kc":1.05, "growing_days":120, "fert":"14-35-14",      "latitude":38},
    "red_onion":        {"N_range":(100,120), "P_range":(40,60),   "K_range":(100,140), "temp":20, "humidity":60, "ph":6.5, "rain":40,  "Kc":1.05, "growing_days":120, "fert":"14-35-14",      "latitude":38},
    "spring_onion":     {"N_range":(80,110),  "P_range":(35,55),   "K_range":(80,120),  "temp":18, "humidity":60, "ph":6.5, "rain":40,  "Kc":0.95, "growing_days":60,  "fert":"Urea",          "latitude":38},
    "shallot":          {"N_range":(90,120),  "P_range":(35,60),   "K_range":(90,130),  "temp":18, "humidity":60, "ph":6.5, "rain":40,  "Kc":1.00, "growing_days":90,  "fert":"14-35-14",      "latitude":40},
    "garlic":           {"N_range":(100,130), "P_range":(40,65),   "K_range":(100,140), "temp":16, "humidity":65, "ph":6.5, "rain":45,  "Kc":1.00, "growing_days":150, "fert":"14-35-14",      "latitude":38},
    "leek":             {"N_range":(120,160), "P_range":(45,70),   "K_range":(110,150), "temp":16, "humidity":70, "ph":6.5, "rain":55,  "Kc":1.05, "growing_days":150, "fert":"Urea",          "latitude":48},
    "chive":            {"N_range":(80,110),  "P_range":(30,55),   "K_range":(80,120),  "temp":18, "humidity":65, "ph":6.5, "rain":50,  "Kc":0.95, "growing_days":90,  "fert":"Compost",       "latitude":45},
    "garlic_chive":     {"N_range":(80,110),  "P_range":(30,55),   "K_range":(80,120),  "temp":20, "humidity":65, "ph":6.5, "rain":50,  "Kc":0.95, "growing_days":90,  "fert":"Compost",       "latitude":35},
    "ramp":             {"N_range":(60,100),  "P_range":(25,50),   "K_range":(70,110),  "temp":12, "humidity":70, "ph":5.5, "rain":70,  "Kc":0.90, "growing_days":90,  "fert":"Compost",       "latitude":42},
    "elephant_garlic":  {"N_range":(100,140), "P_range":(40,65),   "K_range":(100,140), "temp":17, "humidity":65, "ph":6.5, "rain":50,  "Kc":1.00, "growing_days":180, "fert":"14-35-14",      "latitude":38},

    # ===== CUCURBITS (15) =====
    "cucumber":         {"N_range":(100,140), "P_range":(60,80),   "K_range":(140,180), "temp":27, "humidity":80, "ph":6.5, "rain":70,  "Kc":1.10, "growing_days":65,  "fert":"Urea",          "latitude":35},
    "pickling_cucumber":{"N_range":(90,130),  "P_range":(55,75),   "K_range":(130,170), "temp":26, "humidity":78, "ph":6.5, "rain":65,  "Kc":1.05, "growing_days":60,  "fert":"Urea",          "latitude":38},
    "zucchini":         {"N_range":(100,140), "P_range":(50,70),   "K_range":(140,180), "temp":24, "humidity":70, "ph":6.5, "rain":75,  "Kc":1.05, "growing_days":60,  "fert":"Urea",          "latitude":40},
    "yellow_squash":    {"N_range":(100,140), "P_range":(50,70),   "K_range":(140,180), "temp":24, "humidity":70, "ph":6.5, "rain":75,  "Kc":1.05, "growing_days":60,  "fert":"Urea",          "latitude":38},
    "pattypan_squash":  {"N_range":(100,140), "P_range":(50,70),   "K_range":(140,180), "temp":24, "humidity":70, "ph":6.5, "rain":75,  "Kc":1.05, "growing_days":65,  "fert":"Urea",          "latitude":38},
    "butternut_squash": {"N_range":(110,150), "P_range":(55,75),   "K_range":(150,190), "temp":24, "humidity":65, "ph":6.5, "rain":70,  "Kc":1.10, "growing_days":100, "fert":"10-26-26",      "latitude":38},
    "acorn_squash":     {"N_range":(110,150), "P_range":(55,75),   "K_range":(150,190), "temp":24, "humidity":65, "ph":6.5, "rain":70,  "Kc":1.10, "growing_days":100, "fert":"10-26-26",      "latitude":38},
    "spaghetti_squash": {"N_range":(110,150), "P_range":(55,75),   "K_range":(150,190), "temp":24, "humidity":65, "ph":6.5, "rain":70,  "Kc":1.10, "growing_days":100, "fert":"10-26-26",      "latitude":40},
    "pumpkin":          {"N_range":(120,160), "P_range":(60,80),   "K_range":(160,200), "temp":25, "humidity":65, "ph":6.5, "rain":75,  "Kc":1.15, "growing_days":110, "fert":"10-26-26",      "latitude":38},
    "cantaloupe":       {"N_range":(80,120),  "P_range":(40,65),   "K_range":(120,160), "temp":27, "humidity":70, "ph":6.5, "rain":50,  "Kc":1.05, "growing_days":90,  "fert":"14-35-14",      "latitude":35},
    "honeydew":         {"N_range":(80,120),  "P_range":(40,65),   "K_range":(120,160), "temp":27, "humidity":65, "ph":6.5, "rain":45,  "Kc":1.05, "growing_days":90,  "fert":"14-35-14",      "latitude":35},
    "watermelon":       {"N_range":(80,110),  "P_range":(30,55),   "K_range":(100,150), "temp":26, "humidity":75, "ph":6.0, "rain":50,  "Kc":1.10, "growing_days":90,  "fert":"10-26-26",      "latitude":30},
    "bitter_melon":     {"N_range":(80,120),  "P_range":(40,65),   "K_range":(100,140), "temp":27, "humidity":75, "ph":6.5, "rain":80,  "Kc":1.05, "growing_days":70,  "fert":"Urea",          "latitude":20},
    "bottle_gourd":     {"N_range":(80,120),  "P_range":(40,65),   "K_range":(100,140), "temp":28, "humidity":70, "ph":6.5, "rain":70,  "Kc":1.05, "growing_days":90,  "fert":"Urea",          "latitude":22},
    "luffa":            {"N_range":(80,120),  "P_range":(40,65),   "K_range":(100,140), "temp":28, "humidity":75, "ph":6.5, "rain":80,  "Kc":1.05, "growing_days":100, "fert":"Urea",          "latitude":20},

    # ===== LEAFY GREENS (15) =====
    "lettuce_butterhead":{"N_range":(100,140),"P_range":(30,55),   "K_range":(80,120),  "temp":16, "humidity":70, "ph":6.5, "rain":50,  "Kc":1.00, "growing_days":60,  "fert":"Urea",          "latitude":40},
    "lettuce_romaine":  {"N_range":(110,150), "P_range":(30,55),   "K_range":(80,120),  "temp":16, "humidity":70, "ph":6.5, "rain":50,  "Kc":1.00, "growing_days":65,  "fert":"Urea",          "latitude":40},
    "lettuce_iceberg":  {"N_range":(120,160), "P_range":(35,60),   "K_range":(90,130),  "temp":15, "humidity":70, "ph":6.5, "rain":55,  "Kc":1.00, "growing_days":70,  "fert":"Urea",          "latitude":40},
    "spinach":          {"N_range":(100,140), "P_range":(20,40),   "K_range":(60,80),   "temp":15, "humidity":70, "ph":6.5, "rain":50,  "Kc":1.00, "growing_days":50,  "fert":"Urea",          "latitude":45},
    "swiss_chard":      {"N_range":(110,150), "P_range":(30,55),   "K_range":(100,140), "temp":18, "humidity":65, "ph":6.5, "rain":55,  "Kc":1.00, "growing_days":60,  "fert":"Urea",          "latitude":42},
    "arugula":          {"N_range":(80,120),  "P_range":(25,50),   "K_range":(70,110),  "temp":16, "humidity":65, "ph":6.5, "rain":45,  "Kc":0.95, "growing_days":40,  "fert":"Compost",       "latitude":40},
    "endive":           {"N_range":(90,130),  "P_range":(30,55),   "K_range":(80,120),  "temp":16, "humidity":70, "ph":6.5, "rain":55,  "Kc":1.00, "growing_days":90,  "fert":"Compost",       "latitude":45},
    "frisee":           {"N_range":(90,130),  "P_range":(30,55),   "K_range":(80,120),  "temp":16, "humidity":70, "ph":6.5, "rain":55,  "Kc":1.00, "growing_days":90,  "fert":"Compost",       "latitude":45},
    "radicchio":        {"N_range":(90,130),  "P_range":(30,55),   "K_range":(80,120),  "temp":16, "humidity":70, "ph":6.5, "rain":55,  "Kc":1.00, "growing_days":90,  "fert":"Compost",       "latitude":45},
    "water_spinach":    {"N_range":(90,130),  "P_range":(30,55),   "K_range":(80,120),  "temp":28, "humidity":85, "ph":6.0, "rain":120, "Kc":1.10, "growing_days":30,  "fert":"Urea",          "latitude":15},
    "new_zealand_spinach":{"N_range":(80,120),"P_range":(25,50),   "K_range":(70,110),  "temp":22, "humidity":65, "ph":6.5, "rain":55,  "Kc":1.00, "growing_days":70,  "fert":"Urea",          "latitude":-40},
    "chrysanthemum_greens":{"N_range":(90,130),"P_range":(30,55),  "K_range":(80,120),  "temp":18, "humidity":70, "ph":6.5, "rain":60,  "Kc":1.00, "growing_days":45,  "fert":"Compost",       "latitude":35},
    "shiso":            {"N_range":(80,120),  "P_range":(25,50),   "K_range":(70,110),  "temp":22, "humidity":70, "ph":6.5, "rain":65,  "Kc":1.00, "growing_days":60,  "fert":"Compost",       "latitude":35},
    "purslane":         {"N_range":(50,90),   "P_range":(20,45),   "K_range":(50,90),   "temp":26, "humidity":60, "ph":6.5, "rain":40,  "Kc":0.90, "growing_days":50,  "fert":"Compost",       "latitude":30},
    "dandelion_greens": {"N_range":(70,110),  "P_range":(25,50),   "K_range":(70,110),  "temp":14, "humidity":65, "ph":6.0, "rain":55,  "Kc":0.95, "growing_days":60,  "fert":"Compost",       "latitude":45},

    # ===== TROPICAL FRUITS (20) =====
    "banana":           {"N_range":(90,120),  "P_range":(70,90),   "K_range":(300,400), "temp":27, "humidity":80, "ph":6.0, "rain":110, "Kc":1.20, "growing_days":300, "fert":"Urea",          "latitude":12},
    "plantain":         {"N_range":(90,120),  "P_range":(70,90),   "K_range":(300,400), "temp":27, "humidity":80, "ph":6.0, "rain":120, "Kc":1.20, "growing_days":330, "fert":"Urea",          "latitude":10},
    "mango":            {"N_range":(20,50),   "P_range":(15,35),   "K_range":(25,45),   "temp":30, "humidity":50, "ph":6.5, "rain":90,  "Kc":0.90, "growing_days":180, "fert":"Compost",       "latitude":20},
    "papaya":           {"N_range":(30,50),   "P_range":(50,70),   "K_range":(45,55),   "temp":30, "humidity":92, "ph":6.8, "rain":145, "Kc":1.10, "growing_days":270, "fert":"14-35-14",      "latitude":15},
    "pineapple":        {"N_range":(60,100),  "P_range":(10,30),   "K_range":(100,150), "temp":28, "humidity":70, "ph":5.5, "rain":100, "Kc":1.00, "growing_days":540, "fert":"Urea",          "latitude":18},
    "coconut":          {"N_range":(15,40),   "P_range":(10,30),   "K_range":(20,45),   "temp":27, "humidity":92, "ph":6.0, "rain":200, "Kc":1.05, "growing_days":360, "fert":"Urea",          "latitude":10},
    "guava":            {"N_range":(25,55),   "P_range":(20,45),   "K_range":(35,65),   "temp":27, "humidity":70, "ph":6.5, "rain":100, "Kc":0.95, "growing_days":270, "fert":"14-35-14",      "latitude":20},
    "passion_fruit":    {"N_range":(50,90),   "P_range":(30,55),   "K_range":(60,100),  "temp":25, "humidity":70, "ph":6.5, "rain":100, "Kc":1.00, "growing_days":240, "fert":"14-35-14",      "latitude":20},
    "dragon_fruit":     {"N_range":(30,65),   "P_range":(20,45),   "K_range":(40,75),   "temp":28, "humidity":60, "ph":6.5, "rain":60,  "Kc":0.85, "growing_days":365, "fert":"Compost",       "latitude":18},
    "rambutan":         {"N_range":(40,80),   "P_range":(20,45),   "K_range":(50,90),   "temp":28, "humidity":82, "ph":6.0, "rain":200, "Kc":1.10, "growing_days":180, "fert":"Compost",       "latitude":10},
    "lychee":           {"N_range":(35,70),   "P_range":(20,45),   "K_range":(50,90),   "temp":25, "humidity":75, "ph":6.0, "rain":130, "Kc":1.00, "growing_days":180, "fert":"Compost",       "latitude":22},
    "jackfruit":        {"N_range":(30,65),   "P_range":(20,45),   "K_range":(50,90),   "temp":28, "humidity":80, "ph":6.5, "rain":150, "Kc":1.00, "growing_days":365, "fert":"Compost",       "latitude":12},
    "durian":           {"N_range":(40,80),   "P_range":(25,50),   "K_range":(60,100),  "temp":28, "humidity":80, "ph":6.5, "rain":200, "Kc":1.10, "growing_days":365, "fert":"Compost",       "latitude":5},
    "starfruit":        {"N_range":(30,65),   "P_range":(20,45),   "K_range":(50,90),   "temp":27, "humidity":75, "ph":6.5, "rain":120, "Kc":0.95, "growing_days":270, "fert":"Compost",       "latitude":10},
    "sapodilla":        {"N_range":(25,55),   "P_range":(15,40),   "K_range":(40,80),   "temp":28, "humidity":70, "ph":6.5, "rain":100, "Kc":0.90, "growing_days":365, "fert":"Compost",       "latitude":18},
    "cherimoya":        {"N_range":(30,65),   "P_range":(20,45),   "K_range":(50,90),   "temp":22, "humidity":65, "ph":6.5, "rain":80,  "Kc":0.95, "growing_days":270, "fert":"Compost",       "latitude":22},
    "soursop":          {"N_range":(30,65),   "P_range":(20,45),   "K_range":(50,90),   "temp":28, "humidity":75, "ph":6.5, "rain":130, "Kc":1.00, "growing_days":270, "fert":"Compost",       "latitude":12},
    "breadfruit":       {"N_range":(25,55),   "P_range":(15,40),   "K_range":(40,80),   "temp":28, "humidity":80, "ph":6.5, "rain":180, "Kc":1.00, "growing_days":365, "fert":"Compost",       "latitude":8},
    "tamarind":         {"N_range":(15,40),   "P_range":(10,30),   "K_range":(20,50),   "temp":28, "humidity":60, "ph":7.0, "rain":65,  "Kc":0.80, "growing_days":365, "fert":"Compost",       "latitude":18},
    "avocado":          {"N_range":(40,80),   "P_range":(20,45),   "K_range":(60,110),  "temp":23, "humidity":65, "ph":6.5, "rain":100, "Kc":1.00, "growing_days":365, "fert":"Compost",       "latitude":22},

    # ===== TEMPERATE FRUITS (20) =====
    "apple":            {"N_range":(10,30),   "P_range":(130,150), "K_range":(190,210), "temp":17, "humidity":92, "ph":6.0, "rain":110, "Kc":1.05, "growing_days":180, "fert":"DAP",           "latitude":45},
    "pear":             {"N_range":(15,35),   "P_range":(25,50),   "K_range":(60,100),  "temp":16, "humidity":80, "ph":6.5, "rain":90,  "Kc":1.00, "growing_days":150, "fert":"DAP",           "latitude":45},
    "peach":            {"N_range":(20,50),   "P_range":(20,45),   "K_range":(60,100),  "temp":20, "humidity":65, "ph":6.5, "rain":80,  "Kc":1.05, "growing_days":150, "fert":"DAP",           "latitude":38},
    "nectarine":        {"N_range":(20,50),   "P_range":(20,45),   "K_range":(60,100),  "temp":21, "humidity":65, "ph":6.5, "rain":80,  "Kc":1.05, "growing_days":150, "fert":"DAP",           "latitude":38},
    "plum":             {"N_range":(15,45),   "P_range":(20,45),   "K_range":(60,100),  "temp":18, "humidity":70, "ph":6.5, "rain":85,  "Kc":1.00, "growing_days":150, "fert":"DAP",           "latitude":42},
    "apricot":          {"N_range":(15,40),   "P_range":(20,45),   "K_range":(55,95),   "temp":20, "humidity":60, "ph":7.0, "rain":65,  "Kc":0.95, "growing_days":120, "fert":"DAP",           "latitude":38},
    "sweet_cherry":     {"N_range":(15,40),   "P_range":(20,45),   "K_range":(50,90),   "temp":17, "humidity":70, "ph":6.5, "rain":80,  "Kc":1.00, "growing_days":120, "fert":"DAP",           "latitude":45},
    "sour_cherry":      {"N_range":(15,40),   "P_range":(20,45),   "K_range":(50,90),   "temp":15, "humidity":70, "ph":6.5, "rain":80,  "Kc":1.00, "growing_days":120, "fert":"DAP",           "latitude":48},
    "quince":           {"N_range":(10,35),   "P_range":(15,40),   "K_range":(40,80),   "temp":16, "humidity":65, "ph":6.5, "rain":75,  "Kc":0.90, "growing_days":180, "fert":"Compost",       "latitude":40},
    "table_grape":      {"N_range":(20,50),   "P_range":(15,40),   "K_range":(60,100),  "temp":22, "humidity":60, "ph":6.0, "rain":60,  "Kc":0.90, "growing_days":180, "fert":"MOP",           "latitude":38},
    "wine_grape":       {"N_range":(20,45),   "P_range":(120,150), "K_range":(190,210), "temp":20, "humidity":65, "ph":6.0, "rain":65,  "Kc":0.90, "growing_days":180, "fert":"MOP",           "latitude":42},
    "fig":              {"N_range":(15,40),   "P_range":(10,30),   "K_range":(30,60),   "temp":25, "humidity":50, "ph":7.0, "rain":60,  "Kc":0.85, "growing_days":180, "fert":"Compost",       "latitude":38},
    "pomegranate":      {"N_range":(15,35),   "P_range":(15,40),   "K_range":(35,65),   "temp":28, "humidity":40, "ph":7.0, "rain":120, "Kc":0.90, "growing_days":180, "fert":"14-35-14",      "latitude":32},
    "persimmon":        {"N_range":(15,40),   "P_range":(15,40),   "K_range":(40,80),   "temp":20, "humidity":65, "ph":6.5, "rain":90,  "Kc":0.95, "growing_days":270, "fert":"Compost",       "latitude":35},
    "elderberry":       {"N_range":(15,40),   "P_range":(15,40),   "K_range":(40,80),   "temp":16, "humidity":70, "ph":6.0, "rain":80,  "Kc":0.90, "growing_days":180, "fert":"Compost",       "latitude":48},
    "mulberry":         {"N_range":(20,50),   "P_range":(15,40),   "K_range":(40,80),   "temp":22, "humidity":65, "ph":6.5, "rain":80,  "Kc":0.95, "growing_days":180, "fert":"Compost",       "latitude":35},
    "olive":            {"N_range":(15,40),   "P_range":(10,30),   "K_range":(30,60),   "temp":20, "humidity":50, "ph":7.5, "rain":50,  "Kc":0.75, "growing_days":270, "fert":"Compost",       "latitude":38},
    "kiwi":             {"N_range":(50,90),   "P_range":(30,55),   "K_range":(80,130),  "temp":17, "humidity":75, "ph":6.5, "rain":100, "Kc":1.05, "growing_days":270, "fert":"DAP",           "latitude":-40},
    "feijoa":           {"N_range":(20,50),   "P_range":(15,40),   "K_range":(40,80),   "temp":18, "humidity":65, "ph":6.5, "rain":80,  "Kc":0.90, "growing_days":270, "fert":"Compost",       "latitude":-38},
    "date_palm":        {"N_range":(15,45),   "P_range":(10,30),   "K_range":(30,65),   "temp":32, "humidity":30, "ph":7.5, "rain":15,  "Kc":0.90, "growing_days":365, "fert":"Compost",       "latitude":25},

    # ===== BERRIES (15) =====
    "strawberry":       {"N_range":(60,100),  "P_range":(40,60),   "K_range":(120,160), "temp":20, "humidity":70, "ph":6.0, "rain":65,  "Kc":1.05, "growing_days":90,  "fert":"MOP",           "latitude":42},
    "blueberry":        {"N_range":(40,80),   "P_range":(30,55),   "K_range":(60,100),  "temp":18, "humidity":70, "ph":4.8, "rain":80,  "Kc":1.00, "growing_days":120, "fert":"Ammonium_Sulfate","latitude":45},
    "raspberry":        {"N_range":(50,90),   "P_range":(35,60),   "K_range":(80,120),  "temp":17, "humidity":70, "ph":6.2, "rain":75,  "Kc":1.00, "growing_days":150, "fert":"Compost",       "latitude":48},
    "blackberry":       {"N_range":(50,90),   "P_range":(35,60),   "K_range":(80,120),  "temp":18, "humidity":70, "ph":6.0, "rain":75,  "Kc":1.00, "growing_days":150, "fert":"Compost",       "latitude":45},
    "cranberry":        {"N_range":(30,70),   "P_range":(20,45),   "K_range":(50,90),   "temp":14, "humidity":75, "ph":4.5, "rain":90,  "Kc":0.95, "growing_days":270, "fert":"Ammonium_Sulfate","latitude":48},
    "gooseberry":       {"N_range":(40,80),   "P_range":(30,55),   "K_range":(70,110),  "temp":15, "humidity":70, "ph":6.5, "rain":70,  "Kc":1.00, "growing_days":150, "fert":"Compost",       "latitude":52},
    "black_currant":    {"N_range":(50,90),   "P_range":(30,55),   "K_range":(80,120),  "temp":14, "humidity":70, "ph":6.5, "rain":75,  "Kc":1.00, "growing_days":150, "fert":"Compost",       "latitude":52},
    "red_currant":      {"N_range":(45,85),   "P_range":(30,55),   "K_range":(75,115),  "temp":14, "humidity":70, "ph":6.5, "rain":75,  "Kc":1.00, "growing_days":150, "fert":"Compost",       "latitude":52},
    "boysenberry":      {"N_range":(50,90),   "P_range":(35,60),   "K_range":(80,120),  "temp":18, "humidity":70, "ph":6.2, "rain":75,  "Kc":1.00, "growing_days":150, "fert":"Compost",       "latitude":45},
    "loganberry":       {"N_range":(50,90),   "P_range":(35,60),   "K_range":(80,120),  "temp":17, "humidity":70, "ph":6.2, "rain":75,  "Kc":1.00, "growing_days":150, "fert":"Compost",       "latitude":48},
    "cloudberry":       {"N_range":(20,55),   "P_range":(15,40),   "K_range":(40,80),   "temp":10, "humidity":75, "ph":4.5, "rain":80,  "Kc":0.90, "growing_days":120, "fert":"Compost",       "latitude":65},
    "lingonberry":      {"N_range":(20,55),   "P_range":(15,40),   "K_range":(40,80),   "temp":10, "humidity":70, "ph":4.5, "rain":70,  "Kc":0.90, "growing_days":120, "fert":"Compost",       "latitude":62},
    "huckleberry":      {"N_range":(30,70),   "P_range":(20,45),   "K_range":(50,90),   "temp":16, "humidity":70, "ph":5.0, "rain":80,  "Kc":0.95, "growing_days":120, "fert":"Compost",       "latitude":48},
    "goji_berry":       {"N_range":(40,80),   "P_range":(25,50),   "K_range":(60,100),  "temp":18, "humidity":55, "ph":7.0, "rain":50,  "Kc":0.90, "growing_days":150, "fert":"Compost",       "latitude":38},
    "sea_buckthorn":    {"N_range":(25,60),   "P_range":(15,40),   "K_range":(40,80),   "temp":14, "humidity":55, "ph":7.0, "rain":45,  "Kc":0.85, "growing_days":180, "fert":"Compost",       "latitude":50},

    # ===== CITRUS (10) =====
    "navel_orange":     {"N_range":(20,50),   "P_range":(10,30),   "K_range":(25,55),   "temp":23, "humidity":75, "ph":6.5, "rain":110, "Kc":0.90, "growing_days":270, "fert":"Urea",          "latitude":30},
    "valencia_orange":  {"N_range":(22,52),   "P_range":(10,30),   "K_range":(25,55),   "temp":24, "humidity":75, "ph":6.5, "rain":110, "Kc":0.90, "growing_days":270, "fert":"Urea",          "latitude":28},
    "lemon":            {"N_range":(20,50),   "P_range":(10,30),   "K_range":(25,55),   "temp":22, "humidity":70, "ph":6.5, "rain":90,  "Kc":0.90, "growing_days":270, "fert":"Urea",          "latitude":30},
    "lime":             {"N_range":(20,50),   "P_range":(10,30),   "K_range":(25,55),   "temp":26, "humidity":75, "ph":6.5, "rain":100, "Kc":0.90, "growing_days":270, "fert":"Urea",          "latitude":20},
    "grapefruit":       {"N_range":(20,50),   "P_range":(10,30),   "K_range":(30,60),   "temp":24, "humidity":75, "ph":6.5, "rain":100, "Kc":0.95, "growing_days":270, "fert":"Urea",          "latitude":28},
    "mandarin":         {"N_range":(15,45),   "P_range":(10,30),   "K_range":(25,55),   "temp":22, "humidity":72, "ph":6.5, "rain":90,  "Kc":0.90, "growing_days":270, "fert":"Urea",          "latitude":30},
    "tangerine":        {"N_range":(15,45),   "P_range":(10,30),   "K_range":(25,55),   "temp":22, "humidity":72, "ph":6.5, "rain":90,  "Kc":0.90, "growing_days":270, "fert":"Urea",          "latitude":30},
    "pomelo":           {"N_range":(20,50),   "P_range":(10,30),   "K_range":(30,60),   "temp":25, "humidity":75, "ph":6.5, "rain":110, "Kc":0.95, "growing_days":300, "fert":"Urea",          "latitude":22},
    "yuzu":             {"N_range":(15,40),   "P_range":(10,30),   "K_range":(25,55),   "temp":18, "humidity":70, "ph":6.5, "rain":90,  "Kc":0.88, "growing_days":270, "fert":"Compost",       "latitude":35},
    "kumquat":          {"N_range":(15,40),   "P_range":(10,30),   "K_range":(25,55),   "temp":20, "humidity":70, "ph":6.5, "rain":90,  "Kc":0.88, "growing_days":270, "fert":"Compost",       "latitude":28},

    # ===== TREE NUTS (10) =====
    "almond":           {"N_range":(20,55),   "P_range":(15,40),   "K_range":(40,80),   "temp":22, "humidity":50, "ph":7.0, "rain":40,  "Kc":0.90, "growing_days":240, "fert":"DAP",           "latitude":37},
    "walnut":           {"N_range":(15,45),   "P_range":(15,40),   "K_range":(40,80),   "temp":18, "humidity":60, "ph":7.0, "rain":70,  "Kc":0.95, "growing_days":270, "fert":"Compost",       "latitude":40},
    "pecan":            {"N_range":(15,45),   "P_range":(15,40),   "K_range":(40,80),   "temp":24, "humidity":60, "ph":6.5, "rain":80,  "Kc":1.00, "growing_days":270, "fert":"Compost",       "latitude":32},
    "cashew":           {"N_range":(20,55),   "P_range":(10,30),   "K_range":(30,65),   "temp":28, "humidity":60, "ph":6.5, "rain":100, "Kc":0.90, "growing_days":270, "fert":"Compost",       "latitude":12},
    "pistachio":        {"N_range":(15,45),   "P_range":(10,30),   "K_range":(30,65),   "temp":24, "humidity":40, "ph":7.5, "rain":30,  "Kc":0.75, "growing_days":270, "fert":"Compost",       "latitude":35},
    "macadamia":        {"N_range":(15,45),   "P_range":(10,30),   "K_range":(35,70),   "temp":24, "humidity":65, "ph":6.0, "rain":130, "Kc":0.95, "growing_days":270, "fert":"Compost",       "latitude":-25},
    "hazelnut":         {"N_range":(15,45),   "P_range":(15,40),   "K_range":(40,80),   "temp":15, "humidity":65, "ph":6.5, "rain":90,  "Kc":0.90, "growing_days":240, "fert":"Compost",       "latitude":45},
    "chestnut":         {"N_range":(10,35),   "P_range":(10,30),   "K_range":(30,65),   "temp":16, "humidity":65, "ph":6.0, "rain":90,  "Kc":0.90, "growing_days":270, "fert":"Compost",       "latitude":42},
    "brazil_nut":       {"N_range":(10,35),   "P_range":(5,25),    "K_range":(20,55),   "temp":27, "humidity":85, "ph":6.0, "rain":200, "Kc":1.00, "growing_days":365, "fert":"Compost",       "latitude":-5},
    "pine_nut":         {"N_range":(10,30),   "P_range":(5,20),    "K_range":(20,50),   "temp":18, "humidity":50, "ph":6.5, "rain":40,  "Kc":0.75, "growing_days":365, "fert":"Compost",       "latitude":38},

    # ===== HERBS (20) =====
    "sweet_basil":      {"N_range":(80,120),  "P_range":(30,50),   "K_range":(80,120),  "temp":24, "humidity":60, "ph":6.0, "rain":60,  "Kc":1.00, "growing_days":60,  "fert":"Compost",       "latitude":35},
    "thai_basil":       {"N_range":(80,120),  "P_range":(30,50),   "K_range":(80,120),  "temp":26, "humidity":65, "ph":6.0, "rain":70,  "Kc":1.00, "growing_days":60,  "fert":"Compost",       "latitude":20},
    "oregano":          {"N_range":(50,90),   "P_range":(20,45),   "K_range":(60,100),  "temp":20, "humidity":50, "ph":7.0, "rain":40,  "Kc":0.85, "growing_days":90,  "fert":"Compost",       "latitude":38},
    "thyme":            {"N_range":(40,80),   "P_range":(20,45),   "K_range":(60,100),  "temp":18, "humidity":50, "ph":7.0, "rain":40,  "Kc":0.80, "growing_days":90,  "fert":"Compost",       "latitude":40},
    "rosemary":         {"N_range":(40,80),   "P_range":(20,45),   "K_range":(60,100),  "temp":22, "humidity":45, "ph":7.0, "rain":35,  "Kc":0.75, "growing_days":120, "fert":"Compost",       "latitude":38},
    "sage":             {"N_range":(40,80),   "P_range":(20,45),   "K_range":(60,100),  "temp":20, "humidity":50, "ph":7.0, "rain":40,  "Kc":0.80, "growing_days":90,  "fert":"Compost",       "latitude":42},
    "marjoram":         {"N_range":(50,90),   "P_range":(20,45),   "K_range":(60,100),  "temp":20, "humidity":50, "ph":7.0, "rain":40,  "Kc":0.85, "growing_days":80,  "fert":"Compost",       "latitude":40},
    "flat_parsley":     {"N_range":(80,120),  "P_range":(30,55),   "K_range":(80,120),  "temp":18, "humidity":65, "ph":6.5, "rain":55,  "Kc":0.95, "growing_days":90,  "fert":"Compost",       "latitude":42},
    "curly_parsley":    {"N_range":(80,120),  "P_range":(30,55),   "K_range":(80,120),  "temp":18, "humidity":65, "ph":6.5, "rain":55,  "Kc":0.95, "growing_days":90,  "fert":"Compost",       "latitude":42},
    "cilantro":         {"N_range":(50,80),   "P_range":(20,40),   "K_range":(40,60),   "temp":20, "humidity":60, "ph":6.5, "rain":40,  "Kc":0.90, "growing_days":45,  "fert":"Compost",       "latitude":30},
    "dill":             {"N_range":(60,100),  "P_range":(25,50),   "K_range":(60,100),  "temp":18, "humidity":60, "ph":6.5, "rain":50,  "Kc":0.90, "growing_days":70,  "fert":"Compost",       "latitude":45},
    "chervil":          {"N_range":(60,100),  "P_range":(25,50),   "K_range":(60,100),  "temp":15, "humidity":65, "ph":6.5, "rain":50,  "Kc":0.90, "growing_days":60,  "fert":"Compost",       "latitude":48},
    "tarragon":         {"N_range":(50,90),   "P_range":(20,45),   "K_range":(60,100),  "temp":18, "humidity":60, "ph":7.0, "rain":45,  "Kc":0.85, "growing_days":90,  "fert":"Compost",       "latitude":45},
    "lavender":         {"N_range":(30,65),   "P_range":(15,40),   "K_range":(40,80),   "temp":20, "humidity":45, "ph":7.0, "rain":35,  "Kc":0.75, "growing_days":120, "fert":"Compost",       "latitude":43},
    "lemon_balm":       {"N_range":(50,90),   "P_range":(20,45),   "K_range":(60,100),  "temp":18, "humidity":60, "ph":7.0, "rain":55,  "Kc":0.85, "growing_days":90,  "fert":"Compost",       "latitude":45},
    "peppermint":       {"N_range":(100,140), "P_range":(30,50),   "K_range":(80,120),  "temp":18, "humidity":65, "ph":7.0, "rain":80,  "Kc":0.95, "growing_days":90,  "fert":"Compost",       "latitude":45},
    "spearmint":        {"N_range":(100,140), "P_range":(30,50),   "K_range":(80,120),  "temp":18, "humidity":65, "ph":7.0, "rain":80,  "Kc":0.95, "growing_days":90,  "fert":"Compost",       "latitude":45},
    "fennel_herb":      {"N_range":(60,100),  "P_range":(25,50),   "K_range":(70,110),  "temp":20, "humidity":55, "ph":6.5, "rain":45,  "Kc":0.90, "growing_days":90,  "fert":"Compost",       "latitude":40},
    "borage":           {"N_range":(40,80),   "P_range":(20,45),   "K_range":(50,90),   "temp":18, "humidity":60, "ph":7.0, "rain":50,  "Kc":0.85, "growing_days":60,  "fert":"Compost",       "latitude":45},
    "chamomile":        {"N_range":(30,65),   "P_range":(15,40),   "K_range":(40,80),   "temp":16, "humidity":55, "ph":6.5, "rain":45,  "Kc":0.80, "growing_days":60,  "fert":"Compost",       "latitude":48},

    # ===== SPICES (15) =====
    "ginger":           {"N_range":(80,120),  "P_range":(35,60),   "K_range":(90,140),  "temp":26, "humidity":75, "ph":6.0, "rain":150, "Kc":1.05, "growing_days":270, "fert":"Compost",       "latitude":20},
    "turmeric":         {"N_range":(80,120),  "P_range":(35,60),   "K_range":(80,130),  "temp":26, "humidity":75, "ph":6.5, "rain":150, "Kc":1.05, "growing_days":270, "fert":"Compost",       "latitude":18},
    "cardamom":         {"N_range":(40,80),   "P_range":(20,45),   "K_range":(60,100),  "temp":22, "humidity":80, "ph":6.0, "rain":200, "Kc":1.00, "growing_days":365, "fert":"Compost",       "latitude":12},
    "vanilla":          {"N_range":(30,65),   "P_range":(20,45),   "K_range":(50,90),   "temp":26, "humidity":85, "ph":6.5, "rain":200, "Kc":1.00, "growing_days":365, "fert":"Compost",       "latitude":12},
    "cinnamon":         {"N_range":(30,65),   "P_range":(15,40),   "K_range":(40,80),   "temp":27, "humidity":80, "ph":6.5, "rain":180, "Kc":1.00, "growing_days":365, "fert":"Compost",       "latitude":8},
    "clove":            {"N_range":(40,80),   "P_range":(15,40),   "K_range":(50,90),   "temp":25, "humidity":80, "ph":6.5, "rain":180, "Kc":1.00, "growing_days":365, "fert":"Compost",       "latitude":5},
    "coriander_seed":   {"N_range":(50,80),   "P_range":(20,40),   "K_range":(40,60),   "temp":22, "humidity":55, "ph":6.5, "rain":45,  "Kc":0.90, "growing_days":90,  "fert":"Compost",       "latitude":30},
    "cumin":            {"N_range":(40,75),   "P_range":(20,40),   "K_range":(30,55),   "temp":25, "humidity":45, "ph":7.0, "rain":35,  "Kc":0.85, "growing_days":100, "fert":"Compost",       "latitude":28},
    "fenugreek":        {"N_range":(20,50),   "P_range":(20,45),   "K_range":(20,45),   "temp":22, "humidity":55, "ph":7.0, "rain":40,  "Kc":0.88, "growing_days":90,  "fert":"Compost",       "latitude":25},
    "star_anise":       {"N_range":(20,55),   "P_range":(10,35),   "K_range":(30,65),   "temp":24, "humidity":75, "ph":6.5, "rain":120, "Kc":0.90, "growing_days":365, "fert":"Compost",       "latitude":22},
    "lemongrass":       {"N_range":(80,130),  "P_range":(30,60),   "K_range":(80,130),  "temp":28, "humidity":75, "ph":6.5, "rain":120, "Kc":1.00, "growing_days":180, "fert":"Compost",       "latitude":12},
    "galangal":         {"N_range":(60,100),  "P_range":(25,50),   "K_range":(70,110),  "temp":27, "humidity":80, "ph":6.0, "rain":150, "Kc":1.00, "growing_days":270, "fert":"Compost",       "latitude":10},
    "wasabi":           {"N_range":(60,100),  "P_range":(30,55),   "K_range":(60,100),  "temp":15, "humidity":80, "ph":6.5, "rain":180, "Kc":1.00, "growing_days":540, "fert":"Compost",       "latitude":35},
    "saffron":          {"N_range":(20,55),   "P_range":(15,40),   "K_range":(40,80),   "temp":15, "humidity":50, "ph":7.0, "rain":45,  "Kc":0.75, "growing_days":210, "fert":"Compost",       "latitude":38},
    "allspice":         {"N_range":(20,55),   "P_range":(10,35),   "K_range":(30,65),   "temp":26, "humidity":75, "ph":6.5, "rain":130, "Kc":0.90, "growing_days":365, "fert":"Compost",       "latitude":18},

    # ===== OIL CROPS (10) =====
    "sunflower":        {"N_range":(80,120),  "P_range":(30,55),   "K_range":(70,110),  "temp":24, "humidity":55, "ph":6.5, "rain":60,  "Kc":1.10, "growing_days":110, "fert":"DAP",           "latitude":38},
    "canola":           {"N_range":(100,150), "P_range":(40,65),   "K_range":(50,90),   "temp":15, "humidity":60, "ph":6.5, "rain":60,  "Kc":1.05, "growing_days":120, "fert":"Urea",          "latitude":48},
    "sesame":           {"N_range":(40,80),   "P_range":(20,45),   "K_range":(30,65),   "temp":28, "humidity":55, "ph":6.5, "rain":40,  "Kc":0.90, "growing_days":120, "fert":"DAP",           "latitude":20},
    "flaxseed":         {"N_range":(40,80),   "P_range":(20,45),   "K_range":(30,65),   "temp":16, "humidity":60, "ph":6.5, "rain":55,  "Kc":0.95, "growing_days":110, "fert":"DAP",           "latitude":50},
    "castor_bean":      {"N_range":(60,100),  "P_range":(25,50),   "K_range":(40,80),   "temp":28, "humidity":55, "ph":6.5, "rain":55,  "Kc":1.00, "growing_days":180, "fert":"Urea",          "latitude":22},
    "safflower":        {"N_range":(40,80),   "P_range":(20,45),   "K_range":(30,65),   "temp":26, "humidity":45, "ph":7.0, "rain":35,  "Kc":0.90, "growing_days":120, "fert":"DAP",           "latitude":30},
    "hemp_seed":        {"N_range":(60,100),  "P_range":(25,50),   "K_range":(50,90),   "temp":20, "humidity":60, "ph":7.0, "rain":60,  "Kc":1.00, "growing_days":120, "fert":"Urea",          "latitude":45},
    "jojoba":           {"N_range":(15,40),   "P_range":(10,30),   "K_range":(20,50),   "temp":28, "humidity":30, "ph":7.0, "rain":25,  "Kc":0.70, "growing_days":365, "fert":"Compost",       "latitude":25},
    "camelina":         {"N_range":(50,90),   "P_range":(20,45),   "K_range":(30,65),   "temp":15, "humidity":55, "ph":6.5, "rain":45,  "Kc":0.90, "growing_days":100, "fert":"DAP",           "latitude":50},
    "oil_palm":         {"N_range":(80,130),  "P_range":(30,60),   "K_range":(100,160), "temp":27, "humidity":80, "ph":5.5, "rain":200, "Kc":1.10, "growing_days":365, "fert":"MOP",           "latitude":5},

    # ===== FIBER/CASH CROPS (10) =====
    "cotton":           {"N_range":(100,140), "P_range":(35,55),   "K_range":(35,55),   "temp":28, "humidity":77, "ph":6.5, "rain":80,  "Kc":1.15, "growing_days":180, "fert":"10-26-26",      "latitude":28},
    "jute":             {"N_range":(60,90),   "P_range":(35,55),   "K_range":(35,55),   "temp":27, "humidity":80, "ph":6.8, "rain":175, "Kc":1.10, "growing_days":120, "fert":"Urea",          "latitude":22},
    "hemp_fiber":       {"N_range":(80,130),  "P_range":(30,60),   "K_range":(50,100),  "temp":20, "humidity":65, "ph":7.0, "rain":60,  "Kc":1.05, "growing_days":120, "fert":"Urea",          "latitude":45},
    "arabica_coffee":   {"N_range":(80,120),  "P_range":(20,40),   "K_range":(25,45),   "temp":20, "humidity":60, "ph":6.0, "rain":170, "Kc":1.00, "growing_days":365, "fert":"14-35-14",      "latitude":10},
    "robusta_coffee":   {"N_range":(90,130),  "P_range":(25,45),   "K_range":(30,55),   "temp":26, "humidity":70, "ph":6.0, "rain":200, "Kc":1.05, "growing_days":365, "fert":"14-35-14",      "latitude":8},
    "green_tea":        {"N_range":(100,150), "P_range":(20,45),   "K_range":(30,60),   "temp":18, "humidity":70, "ph":5.5, "rain":150, "Kc":1.00, "growing_days":365, "fert":"Ammonium_Sulfate","latitude":30},
    "cacao":            {"N_range":(50,90),   "P_range":(25,50),   "K_range":(50,100),  "temp":27, "humidity":80, "ph":6.5, "rain":200, "Kc":1.05, "growing_days":365, "fert":"Compost",       "latitude":8},
    "tobacco":          {"N_range":(80,130),  "P_range":(30,60),   "K_range":(60,110),  "temp":24, "humidity":65, "ph":6.5, "rain":75,  "Kc":1.05, "growing_days":90,  "fert":"Calcium_Nitrate","latitude":30},
    "rubber_tree":      {"N_range":(30,70),   "P_range":(15,40),   "K_range":(30,70),   "temp":27, "humidity":80, "ph":5.5, "rain":200, "Kc":1.00, "growing_days":365, "fert":"Urea",          "latitude":8},
    "sugarcane":        {"N_range":(100,160), "P_range":(40,70),   "K_range":(80,140),  "temp":28, "humidity":75, "ph":6.5, "rain":150, "Kc":1.25, "growing_days":365, "fert":"Urea",          "latitude":18},

    # ===== MEDICINAL PLANTS (15) =====
    "aloe_vera":        {"N_range":(20,50),   "P_range":(10,30),   "K_range":(20,50),   "temp":25, "humidity":40, "ph":7.0, "rain":30,  "Kc":0.60, "growing_days":365, "fert":"Compost",       "latitude":25},
    "echinacea":        {"N_range":(30,65),   "P_range":(15,40),   "K_range":(30,65),   "temp":18, "humidity":60, "ph":7.0, "rain":60,  "Kc":0.80, "growing_days":180, "fert":"Compost",       "latitude":42},
    "american_ginseng": {"N_range":(25,55),   "P_range":(15,40),   "K_range":(30,65),   "temp":14, "humidity":70, "ph":5.5, "rain":90,  "Kc":0.85, "growing_days":365, "fert":"Compost",       "latitude":42},
    "ashwagandha":      {"N_range":(30,65),   "P_range":(15,40),   "K_range":(25,60),   "temp":25, "humidity":45, "ph":7.5, "rain":40,  "Kc":0.75, "growing_days":180, "fert":"Compost",       "latitude":22},
    "moringa":          {"N_range":(30,70),   "P_range":(15,40),   "K_range":(30,70),   "temp":28, "humidity":60, "ph":7.0, "rain":80,  "Kc":0.90, "growing_days":240, "fert":"Compost",       "latitude":18},
    "st_johns_wort":    {"N_range":(25,60),   "P_range":(15,40),   "K_range":(25,60),   "temp":18, "humidity":55, "ph":7.0, "rain":60,  "Kc":0.80, "growing_days":90,  "fert":"Compost",       "latitude":45},
    "valerian":         {"N_range":(40,80),   "P_range":(20,45),   "K_range":(40,80),   "temp":16, "humidity":65, "ph":7.0, "rain":75,  "Kc":0.85, "growing_days":365, "fert":"Compost",       "latitude":50},
    "milk_thistle":     {"N_range":(25,60),   "P_range":(15,40),   "K_range":(25,60),   "temp":20, "humidity":55, "ph":7.0, "rain":50,  "Kc":0.80, "growing_days":120, "fert":"Compost",       "latitude":40},
    "feverfew":         {"N_range":(25,55),   "P_range":(15,40),   "K_range":(25,55),   "temp":16, "humidity":60, "ph":7.0, "rain":60,  "Kc":0.80, "growing_days":90,  "fert":"Compost",       "latitude":48},
    "goldenseal":       {"N_range":(25,55),   "P_range":(15,40),   "K_range":(25,55),   "temp":15, "humidity":70, "ph":6.0, "rain":90,  "Kc":0.85, "growing_days":180, "fert":"Compost",       "latitude":42},
    "astragalus":       {"N_range":(20,50),   "P_range":(10,35),   "K_range":(20,55),   "temp":18, "humidity":55, "ph":7.0, "rain":50,  "Kc":0.80, "growing_days":180, "fert":"Compost",       "latitude":40},
    "rhodiola":         {"N_range":(15,40),   "P_range":(10,30),   "K_range":(15,40),   "temp":10, "humidity":60, "ph":6.5, "rain":55,  "Kc":0.75, "growing_days":365, "fert":"Compost",       "latitude":55},
    "stinging_nettle":  {"N_range":(50,90),   "P_range":(20,45),   "K_range":(50,90),   "temp":16, "humidity":70, "ph":7.0, "rain":80,  "Kc":0.90, "growing_days":90,  "fert":"Compost",       "latitude":50},
    "skullcap":         {"N_range":(25,55),   "P_range":(15,40),   "K_range":(25,55),   "temp":18, "humidity":65, "ph":7.0, "rain":70,  "Kc":0.80, "growing_days":90,  "fert":"Compost",       "latitude":42},
    "witch_hazel":      {"N_range":(15,40),   "P_range":(10,30),   "K_range":(20,50),   "temp":15, "humidity":65, "ph":6.0, "rain":80,  "Kc":0.80, "growing_days":365, "fert":"Compost",       "latitude":42},

    # ===== ORNAMENTAL (10) =====
    "hybrid_rose":      {"N_range":(100,140), "P_range":(50,80),   "K_range":(100,150), "temp":20, "humidity":60, "ph":6.5, "rain":80,  "Kc":1.00, "growing_days":365, "fert":"14-35-14",      "latitude":40},
    "tulip":            {"N_range":(60,100),  "P_range":(50,80),   "K_range":(100,150), "temp":12, "humidity":65, "ph":7.0, "rain":60,  "Kc":0.90, "growing_days":90,  "fert":"14-35-14",      "latitude":52},
    "chrysanthemum":    {"N_range":(100,140), "P_range":(50,80),   "K_range":(100,150), "temp":18, "humidity":65, "ph":6.5, "rain":70,  "Kc":1.00, "growing_days":120, "fert":"14-35-14",      "latitude":38},
    "dahlia":           {"N_range":(80,120),  "P_range":(60,90),   "K_range":(100,150), "temp":20, "humidity":65, "ph":6.5, "rain":70,  "Kc":1.00, "growing_days":120, "fert":"14-35-14",      "latitude":45},
    "african_marigold": {"N_range":(60,100),  "P_range":(40,70),   "K_range":(80,120),  "temp":22, "humidity":60, "ph":6.5, "rain":55,  "Kc":0.95, "growing_days":90,  "fert":"Compost",       "latitude":35},
    "zinnia":           {"N_range":(50,90),   "P_range":(40,70),   "K_range":(70,110),  "temp":24, "humidity":60, "ph":6.5, "rain":50,  "Kc":0.90, "growing_days":60,  "fert":"Compost",       "latitude":35},
    "asiatic_lily":     {"N_range":(80,120),  "P_range":(50,80),   "K_range":(90,130),  "temp":18, "humidity":65, "ph":6.5, "rain":70,  "Kc":0.95, "growing_days":90,  "fert":"14-35-14",      "latitude":45},
    "moth_orchid":      {"N_range":(30,65),   "P_range":(30,60),   "K_range":(30,65),   "temp":24, "humidity":75, "ph":6.5, "rain":80,  "Kc":0.80, "growing_days":365, "fert":"Vermicompost",  "latitude":20},
    "carnation":        {"N_range":(80,120),  "P_range":(50,80),   "K_range":(90,130),  "temp":18, "humidity":65, "ph":6.5, "rain":65,  "Kc":1.00, "growing_days":180, "fert":"14-35-14",      "latitude":42},
    "gerbera_daisy":    {"N_range":(80,120),  "P_range":(50,80),   "K_range":(90,130),  "temp":22, "humidity":65, "ph":6.5, "rain":65,  "Kc":1.00, "growing_days":120, "fert":"14-35-14",      "latitude":35},

    # ===== FORAGE (10) =====
    "alfalfa":          {"N_range":(20,50),   "P_range":(35,60),   "K_range":(100,160), "temp":20, "humidity":55, "ph":7.0, "rain":50,  "Kc":1.05, "growing_days":60,  "fert":"SSP",           "latitude":40},
    "red_clover":       {"N_range":(15,40),   "P_range":(30,55),   "K_range":(80,130),  "temp":18, "humidity":65, "ph":6.5, "rain":65,  "Kc":1.00, "growing_days":60,  "fert":"SSP",           "latitude":48},
    "white_clover":     {"N_range":(10,35),   "P_range":(25,50),   "K_range":(70,120),  "temp":17, "humidity":65, "ph":6.5, "rain":65,  "Kc":0.95, "growing_days":60,  "fert":"SSP",           "latitude":50},
    "hairy_vetch":      {"N_range":(10,35),   "P_range":(20,45),   "K_range":(50,90),   "temp":15, "humidity":65, "ph":6.5, "rain":60,  "Kc":0.95, "growing_days":90,  "fert":"SSP",           "latitude":45},
    "annual_ryegrass":  {"N_range":(80,130),  "P_range":(30,55),   "K_range":(70,120),  "temp":15, "humidity":70, "ph":6.5, "rain":70,  "Kc":1.00, "growing_days":60,  "fert":"Urea",          "latitude":50},
    "bermuda_grass":    {"N_range":(100,160), "P_range":(35,60),   "K_range":(80,130),  "temp":26, "humidity":60, "ph":6.5, "rain":65,  "Kc":0.95, "growing_days":60,  "fert":"Urea",          "latitude":30},
    "timothy_grass":    {"N_range":(80,130),  "P_range":(30,55),   "K_range":(70,120),  "temp":15, "humidity":70, "ph":6.5, "rain":70,  "Kc":1.00, "growing_days":90,  "fert":"Urea",          "latitude":50},
    "tall_fescue":      {"N_range":(90,140),  "P_range":(30,55),   "K_range":(70,120),  "temp":18, "humidity":65, "ph":6.5, "rain":70,  "Kc":1.00, "growing_days":90,  "fert":"Urea",          "latitude":45},
    "bahia_grass":      {"N_range":(80,130),  "P_range":(25,50),   "K_range":(60,110),  "temp":26, "humidity":65, "ph":6.0, "rain":80,  "Kc":0.95, "growing_days":60,  "fert":"Urea",          "latitude":28},
    "sudangrass":       {"N_range":(90,140),  "P_range":(30,55),   "K_range":(70,120),  "temp":27, "humidity":60, "ph":6.5, "rain":70,  "Kc":1.05, "growing_days":90,  "fert":"Urea",          "latitude":25},
}

# ---------------------------------------------------------------------------
# VARIETY GENERATION
# ---------------------------------------------------------------------------
VARIETY_SUFFIXES = ["early", "mid", "late", "dwarf", "giant"]

def _generate_variety(base_name: str, base_spec: Dict[str, Any], suffix: str) -> tuple:
    """Generate a variety with ±15% parameter variation."""
    rng = 0.15
    def vary_range(lo, hi):
        scale = random.uniform(1 - rng, 1 + rng)
        new_lo = max(0.0, lo * scale)
        new_hi = max(new_lo + 1.0, hi * scale)
        return (round(new_lo, 1), round(new_hi, 1))

    def vary_scalar(v, mn=None, mx=None):
        result = v * random.uniform(1 - rng, 1 + rng)
        if mn is not None:
            result = max(mn, result)
        if mx is not None:
            result = min(mx, result)
        return round(result, 4)

    # Suffix-specific biases
    bias = 1.0
    gd_bias = 1.0
    kc_bias = 1.0
    if suffix == "early":
        gd_bias = 0.85
        bias = 0.92
    elif suffix == "late":
        gd_bias = 1.15
        bias = 1.08
    elif suffix == "dwarf":
        bias = 0.88
        gd_bias = 0.90
        kc_bias = 0.92
    elif suffix == "giant":
        bias = 1.12
        gd_bias = 1.10
        kc_bias = 1.08

    new_spec = {
        "N_range": vary_range(base_spec["N_range"][0] * bias, base_spec["N_range"][1] * bias),
        "P_range": vary_range(base_spec["P_range"][0] * bias, base_spec["P_range"][1] * bias),
        "K_range": vary_range(base_spec["K_range"][0] * bias, base_spec["K_range"][1] * bias),
        "temp": vary_scalar(base_spec["temp"], 0.0, 45.0),
        "humidity": vary_scalar(base_spec["humidity"], 10.0, 99.0),
        "ph": round(base_spec["ph"] * random.uniform(1 - 0.05, 1 + 0.05), 2),
        "rain": vary_scalar(base_spec["rain"], 5.0),
        "Kc": round(base_spec["Kc"] * kc_bias * random.uniform(1 - 0.10, 1 + 0.10), 3),
        "growing_days": max(15, int(base_spec["growing_days"] * gd_bias * random.uniform(0.90, 1.10))),
        "fert": base_spec["fert"],
        "latitude": base_spec["latitude"],
    }
    variety_name = f"{base_name}_{suffix}"
    return variety_name, new_spec


def build_all_species() -> Dict[str, Any]:
    """Build complete species dict: base + 5 varieties each."""
    all_species = dict(BASE_SPECIES)
    for base_name, base_spec in BASE_SPECIES.items():
        for suffix in VARIETY_SUFFIXES:
            vname, vspec = _generate_variety(base_name, base_spec, suffix)
            all_species[vname] = vspec
    return all_species


# ---------------------------------------------------------------------------
# PENMAN-MONTEITH ET0
# ---------------------------------------------------------------------------
def compute_penman_monteith_et0(temp: float, humidity: float, wind_speed: float = 2.0) -> float:
    """FAO-56 Penman-Monteith daily ET0 (mm/day)."""
    es = 0.6108 * math.exp(17.27 * temp / (temp + 237.3))
    ea = es * (humidity / 100.0)
    vpd = max(0.0, es - ea)
    delta = 4098 * es / ((temp + 237.3) ** 2)
    P_atm = 101.325
    gamma = 0.000665 * P_atm
    Rn = max(0.5, 0.408 * (0.77 * (0.25 + 0.5) * (0.082 * temp + 0.5)))
    numerator = 0.408 * delta * Rn + gamma * (900 / (temp + 273)) * wind_speed * vpd
    denominator = delta + gamma * (1 + 0.34 * wind_speed)
    return max(0.5, numerator / denominator)


def compute_water_requirement(temp: float, humidity: float, rain: float,
                               Kc: float, growing_days: int) -> float:
    """FAO-56 net seasonal irrigation requirement (mm)."""
    ET0 = compute_penman_monteith_et0(temp, humidity)
    ETc = ET0 * Kc * growing_days
    # USDA effective rainfall formula
    if rain <= 250:
        eff_rain = rain * (125 - 0.2 * rain) / 125
    else:
        eff_rain = 125 + 0.1 * rain
    return max(50.0, ETc - max(0, eff_rain))


# ---------------------------------------------------------------------------
# FERTILIZER SELECTION
# ---------------------------------------------------------------------------
def select_fertilizer(N: float, P: float, K: float,
                       crop_name: str, default_fert: str) -> str:
    """Select fertilizer based on nutrient balance equations."""
    # Organic crop override
    base_crop = crop_name.rsplit("_", 1)[0] if crop_name.endswith(
        tuple(VARIETY_SUFFIXES)) else crop_name
    if base_crop in ORGANIC_CROPS or crop_name in ORGANIC_CROPS:
        return random.choice(["Compost", "Vermicompost", "Neem_Cake"])

    # Compute deficiency scores using midpoint of typical range
    # We use the actual NPK values relative to crop mean to score deficiency
    # Deficiency score = (expected_mean - actual) / expected_mean
    # Here N/P/K are the row values drawn from the spec range, so deficiency
    # is detected when value is well below the spec mean.
    # We do a simplified check: if value < 0.75 * lower bound → deficient
    n_score = 0.0  # placeholder; actual scoring done against row spec below
    _ = n_score  # unused – we'll just use threshold logic on raw values

    # Use ratio approach: flag deficiency if value is low
    total = N + P + K
    if total == 0:
        return default_fert

    n_frac = N / total
    p_frac = P / total
    k_frac = K / total

    if n_frac < 0.30 and N < 60:
        return random.choice(["Urea", "Ammonium_Sulfate", "Calcium_Nitrate"])
    if p_frac < 0.20 and P < 25:
        return random.choice(["DAP", "SSP", "TSP"])
    if k_frac < 0.25 and K < 30:
        return random.choice(["MOP", "Potassium_Sulfate"])

    return default_fert


# ---------------------------------------------------------------------------
# ROW GENERATOR
# ---------------------------------------------------------------------------
def generate_row(crop_name: str, spec: Dict[str, Any]) -> Dict[str, Any]:
    """Generate one data row for a given crop and spec."""
    N = random.uniform(spec["N_range"][0] - spec["N_range"][0]*0.15,
                       spec["N_range"][1] + spec["N_range"][1]*0.10)
    P = random.uniform(spec["P_range"][0] - spec["P_range"][0]*0.15,
                       spec["P_range"][1] + spec["P_range"][1]*0.10)
    K = random.uniform(spec["K_range"][0] - spec["K_range"][0]*0.15,
                       spec["K_range"][1] + spec["K_range"][1]*0.10)
    N = max(0.0, N)
    P = max(0.0, P)
    K = max(0.0, K)

    temp_base = spec["temp"]
    temp = random.gauss(temp_base, temp_base * 0.12)
    temp = max(0.0, min(50.0, temp))

    hum_base = spec["humidity"]
    humidity = random.gauss(hum_base, hum_base * 0.10)
    humidity = max(10.0, min(99.0, humidity))

    ph = random.gauss(spec["ph"], 0.4)
    ph = round(max(3.5, min(9.5, ph)), 2)

    rain = random.gauss(spec["rain"], spec["rain"] * 0.20)
    rain = max(5.0, rain)

    Kc = spec["Kc"] * random.uniform(0.92, 1.08)
    growing_days = max(15, int(spec["growing_days"] * random.uniform(0.90, 1.10)))

    # Derived features
    N_P_ratio = round(N / P, 3) if P > 0 else 0.0
    N_K_ratio = round(N / K, 3) if K > 0 else 0.0
    P_K_ratio = round(P / K, 3) if K > 0 else 0.0
    total_NPK = round(N + P + K, 2)

    # VPD
    es = 0.6108 * math.exp(17.27 * temp / (temp + 237.3))
    ea = es * (humidity / 100.0)
    vpd = round(max(0.0, es - ea), 4)

    et0 = compute_penman_monteith_et0(temp, humidity)
    water_req = compute_water_requirement(temp, humidity, rain, Kc, growing_days)

    heat_stress_index = round(max(0.0, (temp - 30) / 10), 4)
    cold_stress_index = round(max(0.0, (15 - temp) / 10), 4)
    ph_deviation = round(abs(ph - 6.5), 3)
    aridity_index = round(rain / (temp + 10), 4) if (temp + 10) > 0 else 0.0

    fert = select_fertilizer(N, P, K, crop_name, spec["fert"])

    return {
        "N": round(N, 2),
        "P": round(P, 2),
        "K": round(K, 2),
        "temperature": round(temp, 2),
        "humidity": round(humidity, 2),
        "ph": ph,
        "rainfall": round(rain, 2),
        "soil_type": random.choice(SOIL_TYPES),
        "season": random.choice(SEASONS),
        "region_climate": random.choice(REGIONS),
        "crop_label": crop_name,
        "fertilizer_recommendation": fert,
        "water_requirement_mm": round(water_req, 1),
        "growing_days": growing_days,
        "Kc": round(Kc, 4),
        "N_P_ratio": N_P_ratio,
        "N_K_ratio": N_K_ratio,
        "P_K_ratio": P_K_ratio,
        "total_NPK": total_NPK,
        "vpd": vpd,
        "et0_daily": round(et0, 4),
        "heat_stress_index": heat_stress_index,
        "cold_stress_index": cold_stress_index,
        "ph_deviation": ph_deviation,
        "aridity_index": aridity_index,
        "latitude": spec["latitude"],
    }


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    N_ROWS = 120_000
    print(f"Building species database ...")
    all_species = build_all_species()
    species_list = list(all_species.keys())
    n_species = len(species_list)
    print(f"  Total species (base + varieties): {n_species}")
    print(f"Generating {N_ROWS:,} rows ...")

    rows_per_species = N_ROWS // n_species
    remainder = N_ROWS % n_species

    data: List[Dict[str, Any]] = []
    for i, crop_name in enumerate(species_list):
        spec = all_species[crop_name]
        count = rows_per_species + (1 if i < remainder else 0)
        for _ in range(count):
            data.append(generate_row(crop_name, spec))

    random.shuffle(data)

    df = pd.DataFrame(data)

    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "crop_dataset_full.csv")
    df.to_csv(output_path, index=False)

    # --- Statistics ---
    print(f"\n{'='*60}")
    print(f"DATASET STATISTICS")
    print(f"{'='*60}")
    print(f"Shape:          {df.shape}")
    print(f"Columns:        {list(df.columns)}")
    print(f"\nCrop distribution (top 20):")
    print(df["crop_label"].value_counts().head(20).to_string())
    print(f"\nFertilizer distribution:")
    print(df["fertilizer_recommendation"].value_counts().to_string())
    print(f"\nWater requirement (mm) stats:")
    print(df["water_requirement_mm"].describe().round(2).to_string())
    print(f"\nET0 daily (mm/day) stats:")
    print(df["et0_daily"].describe().round(4).to_string())
    print(f"\nVPD (kPa) stats:")
    print(df["vpd"].describe().round(4).to_string())
    print(f"\nNPK stats:")
    print(df[["N","P","K","total_NPK"]].describe().round(2).to_string())
    print(f"\nSaved to: {output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
