# Kenny Schlegel, Peer Neubert, Peter Protzel
#
# HDC-MiniROCKET : Explicit Time Encoding in Time Series Classification with Hyperdimensional Computing
# Copyright (C) 2022 Chair of Automation Technology / TU Chemnitz

import os

# chose the path concerning the used pc
data_path_univar = os.path.abspath('data/Univariate_ts')

UCR_SETS = [
    data_path_univar + 'ACSF1',  # 0
    data_path_univar + 'Adiac',  # 1
    data_path_univar + 'AllGestureWiimoteX',  # 2
    data_path_univar + 'AllGestureWiimoteY',  # 3
    data_path_univar + 'AllGestureWiimoteZ',  # 4
    data_path_univar + 'ArrowHead',  # 5
    data_path_univar + 'BME',  # 6
    data_path_univar + 'Beef',  # 7
    data_path_univar + 'BeetleFly',  # 8
    data_path_univar + 'BirdChicken',  # 9
    data_path_univar + 'CBF',  # 10
    data_path_univar + 'Car',  # 11
    data_path_univar + 'Chinatown',  # 12
    data_path_univar + 'ChlorineConcentration',  # 13
    data_path_univar + 'CinCECGTorso',  # 14
    data_path_univar + 'Coffee',  # 15
    data_path_univar + 'Computers',  # 16
    data_path_univar + 'CricketX',  # 17
    data_path_univar + 'CricketY',  # 18
    data_path_univar + 'CricketZ',  # 19
    data_path_univar + 'Crop',  # 20
    data_path_univar + 'DiatomSizeReduction',  # 21
    data_path_univar + 'DistalPhalanxOutlineAgeGroup',  # 22
    data_path_univar + 'DistalPhalanxOutlineCorrect',  # 23
    data_path_univar + 'DistalPhalanxTW',  # 24
    data_path_univar + 'DodgerLoopDay',  # 25
    data_path_univar + 'DodgerLoopGame',  # 26
    data_path_univar + 'DodgerLoopWeekend',  # 27
    data_path_univar + 'ECG200',  # 28
    data_path_univar + 'ECG5000',  # 29
    data_path_univar + 'ECGFiveDays',  # 30
    data_path_univar + 'EOGHorizontalSignal',  # 31
    data_path_univar + 'EOGVerticalSignal',  # 32
    data_path_univar + 'Earthquakes',  # 33
    data_path_univar + 'ElectricDevices',  # 34
    data_path_univar + 'EthanolLevel',  # 35
    data_path_univar + 'FaceAll',  # 36
    data_path_univar + 'FaceFour',  # 37
    data_path_univar + 'FacesUCR',  # 38
    data_path_univar + 'FiftyWords',  # 39
    data_path_univar + 'Fish',  # 40
    data_path_univar + 'FordA',  # 41
    data_path_univar + 'FordB',  # 42
    data_path_univar + 'FreezerRegularTrain',  # 43
    data_path_univar + 'FreezerSmallTrain',  # 44
    data_path_univar + 'Fungi',  # 45
    data_path_univar + 'GestureMidAirD1',  # 46
    data_path_univar + 'GestureMidAirD2',  # 47
    data_path_univar + 'GestureMidAirD3',  # 48
    data_path_univar + 'GesturePebbleZ1',  # 49
    data_path_univar + 'GesturePebbleZ2',  # 50
    data_path_univar + 'GunPoint',  # 51
    data_path_univar + 'GunPointAgeSpan',  # 52
    data_path_univar + 'GunPointMaleVersusFemale',  # 53
    data_path_univar + 'GunPointOldVersusYoung',  # 54
    data_path_univar + 'Ham',  # 55
    data_path_univar + 'HandOutlines',  # 56
    data_path_univar + 'Haptics',  # 57
    data_path_univar + 'Herring',  # 58
    data_path_univar + 'HouseTwenty',  # 59
    data_path_univar + 'InlineSkate',  # 60
    data_path_univar + 'InsectEPGRegularTrain',  # 61
    data_path_univar + 'InsectEPGSmallTrain',  # 62
    data_path_univar + 'InsectWingbeatSound',  # 63
    data_path_univar + 'ItalyPowerDemand',  # 64
    data_path_univar + 'LargeKitchenAppliances',  # 65
    data_path_univar + 'Lightning2',  # 66
    data_path_univar + 'Lightning7',  # 67
    data_path_univar + 'Mallat',  # 68
    data_path_univar + 'Meat',  # 69
    data_path_univar + 'MedicalImages',  # 70
    data_path_univar + 'MelbournePedestrian',  # 71
    data_path_univar + 'MiddlePhalanxOutlineAgeGroup',  # 72
    data_path_univar + 'MiddlePhalanxOutlineCorrect',  # 73
    data_path_univar + 'MiddlePhalanxTW',  # 74
    data_path_univar + 'MixedShapesRegularTrain',  # 75
    data_path_univar + 'MixedShapesSmallTrain',  # 76
    data_path_univar + 'MoteStrain',  # 77
    data_path_univar + 'NonInvasiveFetalECGThorax1',  # 78
    data_path_univar + 'NonInvasiveFetalECGThorax2',  # 79
    data_path_univar + 'OSULeaf',  # 80
    data_path_univar + 'OliveOil',  # 81
    data_path_univar + 'PLAID',  # 82
    data_path_univar + 'PhalangesOutlinesCorrect',  # 83
    data_path_univar + 'Phoneme',  # 84
    data_path_univar + 'PickupGestureWiimoteZ',  # 85
    data_path_univar + 'PigAirwayPressure',  # 86
    data_path_univar + 'PigArtPressure',  # 87
    data_path_univar + 'PigCVP',  # 88
    data_path_univar + 'Plane',  # 89
    data_path_univar + 'PowerCons',  # 90
    data_path_univar + 'ProximalPhalanxOutlineAgeGroup',  # 91
    data_path_univar + 'ProximalPhalanxOutlineCorrect',  # 92
    data_path_univar + 'ProximalPhalanxTW',  # 93
    data_path_univar + 'RefrigerationDevices',  # 94
    data_path_univar + 'Rock',  # 95
    data_path_univar + 'ScreenType',  # 96
    data_path_univar + 'SemgHandGenderCh2',  # 97
    data_path_univar + 'SemgHandMovementCh2',  # 98
    data_path_univar + 'SemgHandSubjectCh2',  # 99
    data_path_univar + 'ShakeGestureWiimoteZ',  # 100
    data_path_univar + 'ShapeletSim',  # 101
    data_path_univar + 'ShapesAll',  # 102
    data_path_univar + 'SmallKitchenAppliances',  # 103
    data_path_univar + 'SmoothSubspace',  # 104
    data_path_univar + 'SonyAIBORobotSurface1',  # 105
    data_path_univar + 'SonyAIBORobotSurface2',  # 106
    data_path_univar + 'StarLightCurves',  # 107
    data_path_univar + 'Strawberry',  # 108
    data_path_univar + 'SwedishLeaf',  # 109
    data_path_univar + 'Symbols',  # 110
    data_path_univar + 'SyntheticControl',  # 111
    data_path_univar + 'ToeSegmentation1',  # 112
    data_path_univar + 'ToeSegmentation2',  # 113
    data_path_univar + 'Trace',  # 114
    data_path_univar + 'TwoLeadECG',  # 115
    data_path_univar + 'TwoPatterns',  # 116
    data_path_univar + 'UMD',  # 117
    data_path_univar + 'UWaveGestureLibraryAll',  # 118
    data_path_univar + 'UWaveGestureLibraryX',  # 119
    data_path_univar + 'UWaveGestureLibraryY',  # 120
    data_path_univar + 'UWaveGestureLibraryZ',  # 121
    data_path_univar + 'Wafer',  # 122
    data_path_univar + 'Wine',  # 123
    data_path_univar + 'WordSynonyms',  # 124
    data_path_univar + 'Worms',  # 125
    data_path_univar + 'WormsTwoClass',  # 126
    data_path_univar + 'Yoga'  # 127
    ]

UCR_PREFIX = [
    'ACSF1',
    'Adiac',
    'AllGestureWiimoteX',
    'AllGestureWiimoteY',
    'AllGestureWiimoteZ',
    'ArrowHead',
    'BME',
    'Beef',
    'BeetleFly',
    'BirdChicken',
    'CBF',
    'Car',
    'Chinatown',
    'ChlorineConcentration',
    'CinCECGTorso',
    'Coffee',
    'Computers',
    'CricketX',
    'CricketY',
    'CricketZ',
    'Crop',
    'DiatomSizeReduction',
    'DistalPhalanxOutlineAgeGroup',
    'DistalPhalanxOutlineCorrect',
    'DistalPhalanxTW',
    'DodgerLoopDay',
    'DodgerLoopGame',
    'DodgerLoopWeekend',
    'ECG200',
    'ECG5000',
    'ECGFiveDays',
    'EOGHorizontalSignal',
    'EOGVerticalSignal',
    'Earthquakes',
    'ElectricDevices',
    'EthanolLevel',
    'FaceAll',
    'FaceFour',
    'FacesUCR',
    'FiftyWords',
    'Fish',
    'FordA',
    'FordB',
    'FreezerRegularTrain',
    'FreezerSmallTrain',
    'Fungi',
    'GestureMidAirD1',
    'GestureMidAirD2',
    'GestureMidAirD3',
    'GesturePebbleZ1',
    'GesturePebbleZ2',
    'GunPoint',
    'GunPointAgeSpan',
    'GunPointMaleVersusFemale',
    'GunPointOldVersusYoung',
    'Ham',
    'HandOutlines',
    'Haptics',
    'Herring',
    'HouseTwenty',
    'InlineSkate',
    'InsectEPGRegularTrain',
    'InsectEPGSmallTrain',
    'InsectWingbeatSound',
    'ItalyPowerDemand',
    'LargeKitchenAppliances',
    'Lightning2',
    'Lightning7',
    'Mallat',
    'Meat',
    'MedicalImages',
    'MelbournePedestrian',
    'MiddlePhalanxOutlineAgeGroup',
    'MiddlePhalanxOutlineCorrect',
    'MiddlePhalanxTW',
    'MixedShapesRegularTrain',
    'MixedShapesSmallTrain',
    'MoteStrain',
    'NonInvasiveFetalECGThorax1',
    'NonInvasiveFetalECGThorax2',
    'OSULeaf',
    'OliveOil',
    'PLAID',
    'PhalangesOutlinesCorrect',
    'Phoneme',
    'PickupGestureWiimoteZ',
    'PigAirwayPressure',
    'PigArtPressure',
    'PigCVP',
    'Plane',
    'PowerCons',
    'ProximalPhalanxOutlineAgeGroup',
    'ProximalPhalanxOutlineCorrect',
    'ProximalPhalanxTW',
    'RefrigerationDevices',
    'Rock',
    'ScreenType',
    'SemgHandGenderCh2',
    'SemgHandMovementCh2',
    'SemgHandSubjectCh2',
    'ShakeGestureWiimoteZ',
    'ShapeletSim',
    'ShapesAll',
    'SmallKitchenAppliances',
    'SmoothSubspace',
    'SonyAIBORobotSurface1',
    'SonyAIBORobotSurface2',
    'StarLightCurves',
    'Strawberry',
    'SwedishLeaf',
    'Symbols',
    'SyntheticControl',
    'ToeSegmentation1',
    'ToeSegmentation2',
    'Trace',
    'TwoLeadECG',
    'TwoPatterns',
    'UMD',
    'UWaveGestureLibraryAll',
    'UWaveGestureLibraryX',
    'UWaveGestureLibraryY',
    'UWaveGestureLibraryZ',
    'Wafer',
    'Wine',
    'WordSynonyms',
    'Worms',
    'WormsTwoClass',
    'Yoga'
    ]

