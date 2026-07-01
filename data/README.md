# Data

This release keeps only the raw input required by the main ECO-Waste-PCCM
workbook-completion experiment:

- `data/raw/What_a_Waste_3.0_CITY_Dataset_&_Codebook.xlsx`

The file is the World Bank What a Waste 3.0 city workbook. The project reads the
`City dataset` sheet, builds a typed city-feature matrix, and writes completed
workbook and audit-queue outputs under `outputs/main/`.
