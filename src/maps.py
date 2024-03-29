"""
This file contains maps for attributes
in the NBI survey data. Each map contains
textual description to the codes described
in the NBI coding guide.
"""
from collections import defaultdict

mapDict = defaultdict()

mapDict['CatOwner'] = {
                         1:'State Highway',
                         2:'County Highway',
                         3:'Town Highway',
                         4:'City Highway',
                         21:'Other State',
                         25:'Other Local',
                         27:'Railroad',
                         62:'Bureau of Indian Affiars',
                         69:'Bureau of Land Management',
                         70:'Corps of Engineers'
                     }

mapDict['CatMaterial'] = {
                         1:'Concrete',
                         2:'ConcreteContinuous',
                         3:'Steel',
                         4:'SteelContinuous',
                         5:'PrestressedConcrete',
                         6:'PrestressedConcreteContinuous',
                         7:'Wood',
                         8:'Masonry',
                         9:'Aluminum',
                         0:'Other',
                    }

mapDict['CatToll'] = {
                    1:'TollBridge',
                    2:'OnTollRoad',
                    3:'OnFreeRoad',
                    4:'OnInterstateToll',
                    5:'TollBridgeSegementUnder',
                 }

mapDict['CatWearingSurface'] = {
                        '1':'Monolithic Concrete',
                        '2':'Integral Concrete',
                        '3':'Latex Concrete',
                        '4':'Low Slump Concrete',
                        '5':'Epoxy Overlay',
                        '6':'Bituminous',
                        '7':'Wood or Timber',
                        '8':'Gravel',
                        '9':'Other',
                        '0':'None',
                        'N':'Not applicable'
                }

mapDict['CatDesignLoad'] = {
                         1:'H10',
                         2:'H15',
                         3:'HS15',
                         4:'H20',
                         5:'HS20',
                         6:'HS20Mod',
                         7:'Pedestrian',
                         8:'Railroad',
                         9:'HS25',
                       }

mapDict['CatDeckStructureType'] = {
                         1:'ConcreteCastInPlace',
                         2:'ConcretePrecastPanels',
                         3:'OpenGrating',
                         4:'CloseGrating',
                         5:'SteelPlate',
                         6:'CorrugatedSteel',
                         7:'Aluminum',
                         8:'Wood',
                         9:'Other',
                        }


mapDict['CatTypeOfDesign'] = {
                         1:'Slab',
                         2:'StringerMultiBeam',
                         3:'GirderAndFloor',
                         4:'TeeBeam',
                         5:'BoxBeamMultiple',
                         6:'BoxBeamSingle',
                         7:'Frame',
                         8:'Orthotropic',
                         9:'TrussDeck',
                         10:'TrussThru',
                         11:'ArchDeck',
                         12:'ArchThru',
                         13:'Suspension',
                         14:'StayedGirder',
                         15:'MovableLift',
                         16:'MovableBascule',
                         17:'MovableSwing',
                         18:'Tunnel',
                         19:'Culvert',
                         20:'MixedTypes',
                         21:'SegmentalBoxGirder',
                         22:'ChannelBeam',
                         0:'Other',
                    }

listOfColumns = ['yearBuilt',
                 'averageDailyTraffic',
                 'avgDailyTruckTraffic',
                 'snowfall',
                 'freezethaw',
                 'latitude',
                 'longitude',
                 'skew',
                 'numberOfSpansInMainUnit',
                 'lengthOfMaximumSpan',
                 'structureLength',
                 'bridgeRoadwayWidthCurbToCurb',
                 'operatingRating',
                 'scourCriticalBridges',
                 'lanesOnStructure',
                 'designatedInspectionFrequency',
                 'materialAluminum',
                 'materialConcrete',
                 'materialConcreteContinuous',
                 'materialPrestreesedConcrete',
                 'materialPrestreesedConcreteContinuous',
                 'materialSteel',
                 'materialSteelConitnuous',
                 'materialWood',
                 'tollOnFreeRoad',
                 'tollTollBridge',
                 'designLoadH10',
                 'designLoadH15',
                 'designLoadH20',
                 'designLoadHS15',
                 'designLoadHS20',
                 'designLoadHS20Mod',
                 'designLoadHS25',
                 'designLoadPedestrian',
                 'designLoadRailroad',
                 'designLoadnan',
                 'deckStructureTypenan',
                 'typeOfDesignArchDeck',
                 'typeOfDesignArchThru',
                 'typeOfDesignBoxBeamMultiple',
                 'typeOfDesignBoxBeamSingle',
                 'typeOfDesignChannelBeam',
                 'typeOfDesignFrame',
                 'typeOfDesignGirderAndFloor',
                 'typeOfDesignMovableBascule',
                 'typeOfDesignMovableLift',
                 'typeOfDesignOther',
                 'typeOfDesignSegmentalBoxGirder',
                 'typeOfDesignSlab',
                 'typeOfDesignStringerMultiBeam',
                 'typeOfDesignSuspension',
                 'typeOfDesignTeeBeam',
                 'typeOfDesignTrussDeck',
                 'typeOfDesignTrussThru']
