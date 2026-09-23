# Data Sources

This directory documents the public data sources used in the study:

**Meteorological Signals in Spatiotemporal Prediction of High-Activity Influenza Weeks in the United States: An Exploratory Machine-Learning Study**

Raw copies of the complete source databases are not redistributed in this repository.

## CDC Influenza Surveillance

Weekly influenza virological surveillance data were obtained from the CDC FluView Interactive application:

**National, Regional, and State Level Outpatient Illness and Viral Surveillance**

Source:

CDC FluView Interactive

Study period:

**2010 epidemiological week 40 through 2026 epidemiological week 24**

Geographical coverage:

- 50 U.S. states
- District of Columbia
- Puerto Rico

Variables used included:

- weekly number of specimens tested;
- weekly number of influenza-positive specimens; and
- weekly influenza test positivity.

Weekly influenza test positivity was calculated as the number of influenza-positive specimens divided by the total number of specimens tested.

Access date:

**22 July 2026**

## ERA5-Land Meteorological Data

Meteorological data were obtained from the ECMWF ERA5-Land daily aggregated reanalysis product through Google Earth Engine.

Temporal coverage used for meteorological processing:

**2000–2026**

The period **2000–2009** was used to calculate climatological baseline values.

The period **2010–2026** was used for the primary influenza prediction analyses.

Meteorological variables included:

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

Population-density information was obtained from 2020 U.S. Census Bureau data.

The population-density variable was used as a socio-geographical predictor.

**Before public release, add the exact Census dataset/table name used in the completed analysis.**

## CDC FluSight Forecast Hub

Archived CDC FluSight influenza hospitalization forecasts and corresponding finalized hospitalization target data were used in the post hoc downstream analysis.

The analysis was restricted to observations that could be matched across:

- study area;
- target epidemiological week; and
- forecast horizon.

The FluSight analysis was not used for primary model development.

**Before public release, add the exact release, version, or commit and access date used in the completed analysis.**

## Analysis-Ready Data

The analysis-ready study-area-week dataset was derived from the public data sources described above.

The final study dataset contained observations across 52 U.S. study areas from 2010 to 2026.

Raw copies of the complete CDC, ERA5-Land, Census, or FluSight source databases are not redistributed in this repository.

No individual-level or identifiable participant data were used.
