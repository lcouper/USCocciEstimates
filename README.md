# US Cocci Estimates

This repository accompanies the manuscript 'Coccidioidomycosis incidence across reporting and non-reporting U.S. states: an ensemble modeling study'

The code used in the main analysis is captured in **'HiddenBurden_Analysis.R'** 

Note that executing this script requires three data files:
1) The tract-level predictor dataset for Arizona and California (for training & testing). **PublicData_TractLevel_CAZ_2013on.csv**
2) The county-level predictor dataset for all U.S. counties (for application). **PublicData_CountyLevel_2013on.csv**
3) The CDC's NNDSS annual, county-level coccidioidomycosis dataset (for comparison). **PublicData_NNDSS_2000_2022.csv**

Versions of these data files are included here so that the R script is fully executable. However, we note that several data sources used in this study were obtained through data use agreements and are therefore not publicly shareable. This includes the tract-level surveillance data for Arizona and California, the canine serological testing data, the human electronic health record data, and the CDC's NNDSS data. Where these data exist in the 3 files above, they have been fully randomized. Thus, while the code can be run and verified, the results presented in the manuscript will not be reproduced exactly from these data.

