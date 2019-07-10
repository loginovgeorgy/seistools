TRACES_SAMPLES_FORMAT_DESCRIPTION = {
    1: 'IBM Float',
    2: '32 bit Integer',
    3: '16 bit Integer',
    5: 'IEEE',
    8: '8 bit Integer',
}

TRACES_SAMPLES_FORMAT = {
    1: 'ibm',
    2: 'i',
    3: 'h',
    5: 'f',
    8: 'B',
}

BIN_HEADER_DESCRIPTOR = {
    'Job': {
        'pos': 0,
        'type': 'i',
    },
    'Line': {
        'pos': 4,
        'type': 'i',
    },
    'Reel': {
        'pos': 8,
        'type': 'i',
    },
    'TracePerRecord': {
        'pos': 12,
        'type': 'h',
    },
    'AuxiliaryTracePerRecord': {
        'pos': 14,
        'type': 'h',
    },
    'SampleIntervalReel': {
        'pos': 16,
        'type': 'h',
    },
    'SampleIntervalField': {
        'pos': 18,
        'type': 'h',
    },
    'NumberOfSamples': {
        'pos': 20,
        'type': 'h',
    },
    'NumberOfSamplesForField': {
        'pos': 22,
        'type': 'h',
    },
    'DataSampleFormatCode': {
        'pos': 24,
        'type': 'h',
    },
    'CDPFold': {
        'pos': 26,
        'type': 'h',
    },
    'TraceSortingCode': {
        'pos': 28,
        'type': 'h',
    },
    'VerticalSumCode': {
        'pos': 30,
        'type': 'h',
    },
    'SweepFrequencyStart': {
        'pos': 32,
        'type': 'h',
    },
    'SweepFrequencyEnd': {
        'pos': 34,
        'type': 'h',
    },
    'SweepLength': {
        'pos': 36,
        'type': 'h',
    },
    'SweepTypeCode': {
        'pos': 38,
        'type': 'h',
    },
    'SweepTraceNumber': {
        'pos': 40,
        'type': 'h',
    },
    'SweepTaperLengthStart': {
        'pos': 42,
        'type': 'h',
    },
    'SweepTaperLengthEnd': {
        'pos': 44,
        'type': 'h',
    },
    'TaperType': {
        'pos': 46,
        'type': 'h',
    },
    'CorrelatedDataTraces': {
        'pos': 48,
        'type': 'h',
    },
    'BinaryGain': {
        'pos': 50,
        'type': 'h',
    },
    'AmplitudeRecoveryMethod': {
        'pos': 52,
        'type': 'h',
    },
    'MeasurementSystem': {
        'pos': 54,
        'type': 'h',
    },
    'ImpulseSignalPolarity': {
        'pos': 56,
        'type': 'h',
    },
    'VibratoryPolarityCode': {
        'pos': 58,
        'type': 'h',
    },
    'SgyRevision': {
        'pos': 300,
        'type': 'h',
    },
    'FixedLengthTraceFlag': {
        'pos': 302,
        'type': 'h',
    },
    'NumberOfExtTextualHeaders': {
        'pos': 304,
        'type': 'h',
    },
}

TRACE_HEADER_DESCRIPTOR = {
    'TraceSequenceLine': {
        'pos': 0,
        'type': 'i'
    },
    'TraceSequenceFile': {
        'pos': 4,
        'type': 'i'
    },
    'FieldRecord': {
        'pos': 8,
        'type': 'i'
    },
    'TraceNumber': {
        'pos': 12,
        'type': 'i'
    },
    'EnergySourcePoint': {
        'pos': 16,
        'type': 'i',
    },
    'cdp': {
        'pos': 20,
        'type': 'i',
    },
    'cdpTrace': {
        'pos': 24,
        'type': 'i',
    },
    'TraceIdentificationCode': {
        'pos': 28,
        'type': 'H',
        'descr': {
            0: {
                1: 'Seismic data',
                2: 'Dead',
                3: 'Dummy',
                4: 'Time Break',
                5: 'Uphole',
                6: 'Sweep',
                7: 'Timing',
                8: 'Water Break',
            },
            1: {
                -1: 'Other',
                0: 'Unknown',
                1: 'Seismic data',
                2: 'Dead',
                3: 'Dummy',
                4: 'Time break',
                5: 'Uphole',
                6: 'Sweep',
                7: 'Timing',
                8: 'Waterbreak',
                9: 'Near-field gun signature',
                10: 'Far-field gun signature',
                11: 'Seismic pressure sensor',
                12: 'Multicomponent seismic sensor - Vertical component',
                13: 'Multicomponent seismic sensor - Cross-line component',
                14: 'Multicomponent seismic sensor - In-line component',
                15: 'Rotated multicomponent seismic sensor - Vertical component',
                16: 'Rotated multicomponent seismic sensor - Transverse component',
                17: 'Rotated multicomponent seismic sensor - Radial component',
                18: 'Vibrator reaction mass',
                19: 'Vibrator baseplate',
                20: 'Vibrator estimated ground force',
                21: 'Vibrator reference',
                22: 'Time-velocity pairs',
            }
        }
    },
    'NSummedTraces': {
        'pos': 30,
        'type': 'h',
    },
    'NStackedTraces': {
        'pos': 32,
        'type': 'h',
    },
    'DataUse': {
        'pos': 34,
        'type': 'h',
        'descr': {
            0: {
                1: 'Production',
                2: 'Test',
            },
            1: {
                1: 'Production',
                2: 'Test',
            }
        }
    },
    'offset': {
        'pos': 36,
        'type': 'i',
    },
    'ReceiverGroupElevation':{
            'pos': 40,
            'type': 'i',
        },
    'SourceSurfaceElevation': {
        'pos': 44,
        'type': 'i',
    },
    'SourceDepth': {
        'pos': 48,
        'type': 'i',
    },
    'ReceiverDatumElevation': {
        'pos': 52,
        'type': 'i',
    },
    'SourceDatumElevation': {
        'pos': 56,
        'type': 'i',
    },
    'SourceWaterDepth': {
        'pos': 60,
        'type': 'i',
    },
    'GroupWaterDepth': {
        'pos': 64,
        'type': 'i',
    },
    'ElevationScalar': {
        'pos': 68,
        'type': 'h',
    },
    'SourceGroupScalar': {
        'pos': 70,
        'type': 'h',
    },
    'SourceX': {
        'pos': 72,
        'type': 'i',
    },
    'SourceY': {
        'pos': 76,
        'type': 'i',
    },
    'GroupX': {
        'pos': 80,
        'type': 'i',
    },
    'GroupY': {
        'pos': 84,
        'type': 'i',
    },
    'CoordinateUnits': {
        'pos': 88,
        'type': 'h',
        'descr': {
            1: {
                1: 'Length (meters or feet)',
                2: 'Seconds of arc',
                3: 'Decimal degrees',
                4: 'Degrees, minutes, seconds (DMS)',
            }
        }
    },
    'WeatheringVelocity': {
        'pos': 90,
        'type': 'h',
    },
    'SubWeatheringVelocity': {
        'pos': 92,
        'type': 'h',
    },
    'SourceUpholeTime': {
        'pos': 94,
        'type': 'h',
    },
    'GroupUpholeTime': {
        'pos': 96,
        'type': 'h',
    },
    'SourceStaticCorrection': {
        'pos': 98,
        'type': 'h',
    },
    'GroupStaticCorrection': {
        'pos': 100,
        'type': 'h',
    },
    'TotalStaticApplied': {
        'pos': 102,
        'type': 'h',
    },
    'LagTimeA': {
        'pos': 104,
        'type': 'h',
    },
    'LagTimeB': {
        'pos': 106,
        'type': 'h',
    },
    'DelayRecordingTime': {
        'pos': 108,
        'type': 'h',
    },
    'MuteTimeStart': {
        'pos': 110,
        'type': 'h',
    },
    'MuteTimeEND': {
        'pos': 112,
        'type': 'h',
    },
    'ns': {
        'pos': 114,
        'type': 'H',
    },
    'dt': {
        'pos': 116,
        'type': 'H',
    },
    'GainType': {
        'pos': 119,
        'type': 'h',
        'descr': {
            0: {
                1: 'Fixes',
                2: 'Binary',
                3: 'Floating point',
            },
            1: {
                1: 'Fixes',
                2: 'Binary',
                3: 'Floating point',
            }
        }
    },
    'InstrumentGainConstant': {
        'pos': 120,
        'type': 'h',
    },
    'InstrumentInitialGain': {
        'pos': 122,
        'type': 'h',
    },
    'Correlated': {
        'pos': 124,
        'type': 'h',
        'descr': {
            0: {
                1: 'No',
                2: 'Yes',
            },
            1: {
                1: 'No',
                2: 'Yes',
            }
        }
    },
    'SweepFrequencyStart': {
        'pos': 126,
        'type': 'h',
    },
    'SweepFrequencyEnd': {
        'pos': 128,
        'type': 'h',
    },
    'SweepLength': {
        'pos': 130,
        'type': 'h',
    },
    'SweepType': {
        'pos': 132,
        'type': 'h',
        'descr': {
            0: {
                1: 'linear',
                2: 'parabolic',
                3: 'exponential',
                4: 'other',
            },
            1: {
                1: 'linear',
                2: 'parabolic',
                3: 'exponential',
                4: 'other',
            }
        }
    },
    'SweepTraceTaperLengthStart': {
        'pos': 134,
        'type': 'h',
    },
    'SweepTraceTaperLengthEnd': {
        'pos': 136,
        'type': 'h',
    },
    'TaperType': {
        'pos': 138,
        'type': 'h',
        'descr': {
            0: {
                1: 'linear',
                2: 'cos2c',
                3: 'other',
            },
            1: {
                1: 'linear',
                2: 'cos2c',
                3: 'other',
            }
        }
    },
    'AliasFilterFrequency': {
        'pos': 140,
        'type': 'h',
    },
    'AliasFilterSlope': {
        'pos': 142,
        'type': 'h',
    },
    'NotchFilterFrequency': {
        'pos': 144,
        'type': 'h',
    },
    'NotchFilterSlope': {
        'pos': 146,
        'type': 'h',
    },
    'LowCutFrequency': {
        'pos': 148,
        'type': 'h',
    },
    'HighCutFrequency': {
        'pos': 150,
        'type': 'h',
    },
    'LowCutSlope': {
        'pos': 152,
        'type': 'h',
    },
    'HighCutSlope': {
        'pos': 154,
        'type': 'h',
    },
    'YearDataRecorded': {
        'pos': 156,
        'type': 'h',
    },
    'DayOfYear': {
        'pos': 158,
        'type': 'h',
    },
    'HourOfDay': {
        'pos': 160,
        'type': 'h',
    },
    'MinuteOfHour': {
        'pos': 162,
        'type': 'h',
    },
    'SecondOfMinute': {
        'pos': 164,
        'type': 'h',
    },
    'TimeBaseCode': {
        'pos': 166,
        'type': 'h',
        'descr': {
            0: {
                1: 'Local',
                2: 'GMT',
                3: 'Other',
            },
            1: {
                1: 'Local',
                2: 'GMT',
                3: 'Other',
                4: 'UTC',
            }
        }
    },
    'TraceWeightingFactor': {
        'pos': 168,
        'type': 'h',
    },
    'GeophoneGroupNumberRoll1': {
        'pos': 170,
        'type': 'h',
    },
    'GeophoneGroupNumberFirstTraceOrigField': {
        'pos': 172,
        'type': 'h',
    },
    'GeophoneGroupNumberLastTraceOrigField': {
        'pos': 174,
        'type': 'h',
    },
    'GapSize': {
        'pos': 176,
        'type': 'h',
    },
    'OverTravel': {
        'pos': 178,
        'type': 'h',
        'descr': {
            0: {
                1: 'down (or behind)',
                2: 'up (or ahead)',
                3: 'other',
            },
            1: {
                1: 'down (or behind)',
                2: 'up (or ahead)',
                3: 'other',
            }
        }
    },
    'cdpX': {
        'pos': 180,
        'type': 'i',
    },
    'cdpY': {
        'pos': 184,
        'type': 'i',
    },
    'Inline3D': {
        'pos': 188,
        'type': 'i',
    },
    'CrossLine3D': {
        'pos': 192,
        'type': 'i',
    },
    'ShotPoint': {
        'pos': 192,
        'type': 'i',
    },
    'ShotPointScalar': {
        'pos': 200,
        'type': 'h',
    },
    'TraceValueMeasurementUnit': {
        'pos': 202,
        'type': 'h',
        'descr': {
            1: {
                -1: 'Other',
                0: 'Unknown (should be described in Data Sample Measurement Units Stanza) ',
                1: 'Pascal (Pa)',
                2: 'Volts (V)',
                3: 'Millivolts (v)',
                4: 'Amperes (A)',
                5: 'Meters (m)',
                6: 'Meters Per Second (m/s)',
                7: 'Meters Per Second squared (m/&s2)Other',
                8: 'Newton (N)',
                9: 'Watt (W)',
            }
        }
    },
    'TransductionConstantMantissa': {
        'pos': 204,
        'type': 'i',
    },
    'TransductionConstantPower': {
        'pos': 208,
        'type': 'h',
    },
    'TransductionUnit': {
        'pos': 210,
        'type': 'h',
        'descr': {
            1: {
                -1: 'Other',
                0: 'Unknown (should be described in Data Sample Measurement Units Stanza) ',
                1: 'Pascal (Pa)',
                2: 'Volts (V)',
                3: 'Millivolts (v)',
                4: 'Amperes (A)',
                5: 'Meters (m)',
                6: 'Meters Per Second (m/s)',
                7: 'Meters Per Second squared (m/&s2)Other',
                8: 'Newton (N)',
                9: 'Watt (W)',
            }
        }
    },
    'TraceIdentifier': {
        'pos': 212,
        'type': 'h',
    },
    'ScalarTraceHeader': {
        'pos': 214,
        'type': 'h',
    },
    'SourceType': {
        'pos': 216,
        'type': 'h',
        'descr': {
            1: {
                -1: 'Other (should be described in Source Type/Orientation stanza)',
                0: 'Unknown',
                1: 'Vibratory - Vertical orientation',
                2: 'Vibratory - Cross-line orientation',
                3: 'Vibratory - In-line orientation',
                4: 'Impulsive - Vertical orientation',
                5: 'Impulsive - Cross-line orientation',
                6: 'Impulsive - In-line orientation',
                7: 'Distributed Impulsive - Vertical orientation',
                8: 'Distributed Impulsive - Cross-line orientation',
                9: 'Distributed Impulsive - In-line orientation',
            }
        }
    },
    'SourceEnergyDirectionMantissa': {
        'pos': 218,
        'type': 'i',
    },
    'SourceEnergyDirectionExponent': {
        'pos': 222,
        'type': 'h',
    },
    'SourceMeasurementMantissa': {
        'pos': 224,
        'type': 'i',
    },
    'SourceMeasurementExponent': {
        'pos': 228,
        'type': 'h',
    },
    'SourceMeasurementUnit': {
        'pos': 230,
        'type': 'h',
        'descr': {
            1: {
                -1: 'Other (should be described in Source Measurement Unit stanza)',
                0: 'Unknown',
                1: 'Joule (J)',
                2: 'Kilowatt (kW)',
                3: 'Pascal (Pa)',
                4: 'Bar-meter (Bar-m)',
                5: 'Newton (N)',
                6: 'Kilograms (kg)',
            }
        }
    },
}
