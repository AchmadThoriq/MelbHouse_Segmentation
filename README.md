# MelbHouse_Segmentation
**A simple price segmentation property using K-Means**

Hello my name is Achmad Thoriq Aminullah :)

In this repository you will see simple process of price segmentation property real estate in Melbourne, Australia using K-Means algorithm. Dataset used in this segmentation call Melbourne Housing Snapshot from Kaggle. 

You can download it in :
https://www.kaggle.com/dansbecker/melbourne-housing-snapshot


**Notes on Specific Variables**

**Rooms:** Number of rooms
**Price:** Price in dollars
**Method:** S - property sold; SP - property sold prior; PI - property passed in; PN - sold prior not disclosed; SN - sold not disclosed; NB - no bid; VB - vendor bid; W - withdrawn prior to auction; SA - sold after auction; SS - sold after auction price not disclosed. N/A - price or highest bid not available.
**Type:** br - bedroom(s); h - house,cottage,villa, semi,terrace; u - unit, duplex; t - townhouse; dev site - development site; o res - other residential.
**SellerG:** Real Estate Agent
**Date:** Date sold
**Distance:** Distance from CBD
**Regionname:** General Region (West, North West, North, North east …etc)
**Propertycount:** Number of properties that exist in the suburb.
**Bedroom2 :** Scraped # of Bedrooms (from different source)
**Bathroom:** Number of Bathrooms
**Car:** Number of carspots
**Landsize:** Land Size
**BuildingArea:** Building Size
**CouncilArea:** Governing council for the area

**Process:**
1. Load Dataset
2. Checking null values using method info()
3. Handling missing values using modus for categorical values and mean for numerical values
4. Exploratory Data Analysis(EDA) to understand and get insight form dataset
5. From EDA process will be decided features/input for prediction using K-Means
5. Using K-Means prediction with n_cluster = 3. Number of n_cluster is get from elbow method
6. Using method describe() to find minimum and maximum values from each features

For more explanation you can check comment from each code. Thanks for your attention.
