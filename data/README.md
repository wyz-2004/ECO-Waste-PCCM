# Data

This project uses the raw input required by the ECO-Waste-PCCM
workbook-completion workflow:

- `data/raw/What_a_Waste_3.0_CITY_Dataset_&_Codebook.xlsx`

The file is the World Bank What a Waste 3.0 city workbook. The project reads the
`City dataset` sheet, builds a typed city-feature matrix, and writes completed
workbook and audit-queue outputs under `outputs/main/`.
