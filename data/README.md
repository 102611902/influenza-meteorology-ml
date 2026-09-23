# Data Sources

This directory documents the public data sources used in the study:

**Meteorological Signals in Spatiotemporal Prediction of High-Activity Influenza Weeks in the United States: An Exploratory Machine-Learning Study**

Raw copies of the complete source databases are not redistributed in this repository.

## CDC Influenza Surveillance

Weekly influenza virological surveillance data were obtained from the CDC FluView Interactive application:

**National, Regional, and State Level Outpatient Illness and Viral Surveillance**

Study period:

**2010 epidemiological week 40 through 2026 epidemiological week 24**

Geographical coverage:

- 50 U.S. states;
- District of Columbia; and
- Puerto Rico.

Variables used included:

- weekly number of specimens tested;
- weekly number of influenza-positive specimens; and
- weekly influenza test positivity.

Weekly influenza test positivity was calculated as the number of influenza-positive specimens divided by the total number of specimens tested.

Access date:

**22 July 2026**

## ERA5-Land Meteorological Data

Meteorological data were obtained from the ECMWF ERA5-Land daily aggregated reanalysis product through Google Earth Engine.

Meteorological data covered 2000–2026.

The period **2000–2009** was used to calculate climatological baseline values.

The period **2010–2026** was used for the primary influenza prediction analyses.

Meteorological variables included measures related to:

- air temperature;
- relative humidity;
- absolute humidity;
- surface wind speed;
- surface pressure;
- solar radiation; and
- precipitation.

Daily meteorological data were aggregated to study-area-level weekly values.

Detailed variable construction, lag definitions, and anomaly calculations are described in the manuscript and Supplementary Information.

## U.S. Census Bureau

Population-density information was obtained from 2020 U.S. Census Bureau data and used as a socio-geographical predictor.

Additional source and processing details are provided in the manuscript and Supplementary Information.

## CDC FluSight Forecast Hub

Archived influenza hospitalization forecasts and corresponding finalized hospitalization target data used in the post hoc contextual analysis were obtained from the CDC FluSight Forecast Hub.

The analysis was restricted to observations that could be matched by:

- study area;
- target epidemiological week; and
- forecast horizon.

The FluSight analysis was used only as a post hoc contextual analysis and was not used for primary model development, algorithm selection, hyperparameter optimization, or feature selection.

Additional matching and outcome-definition details are provided in the manuscript and Supplementary Information.

## Analysis-Ready Data

The analysis-ready study-area-week dataset was derived from the publicly available sources described above.

The final study dataset contained observations across 52 U.S. study areas from 2010 to 2026.

Raw copies of the complete CDC, ERA5-Land, Census, and FluSight source databases are not redistributed in this repository.

No individual-level or identifiable participant data were used.
