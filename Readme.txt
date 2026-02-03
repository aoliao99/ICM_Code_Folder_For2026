This folder contains the modeling code for Problem F of the 2026 ICM competition.
Team ID: 2617154
To better present our code files, we have re-uploaded them to this new repository. We apologize for any inconvenience this may cause.
File descriptions are as follows:
The AHP-TOPSIS.py file provides data references for calculating alpha coefficients and proposing recommendations regarding non-employment factors.
Regression_Prediction.py uses data from three occupations over the past decade to fit models, calculating the natural growth rate r and environmental carrying capacity K, providing parameters for the subsequent SD model.
SD.py is the system dynamics model code file, using the parameters obtained above and collected data to predict trends for the three occupations in an AI context.
The SD_Model_for_Specific_Institutions.py file contains the system dynamics model code that incorporates the non-employment factor weights from the AHP-TOPSIS output and the specific data for the three institutions.
The Sensitivity_Analysis.py file contains the code for sensitivity analysis of the system dynamics model.
Spider_Plot.py generates corresponding Spider Plots based on the three key indicator weights from AHP-TOPSIS.
Final_SD_output_image.py integrates the three occupations to produce the final system dynamics model output diagram.

SD_Output_Diagram_for_Specific_Institutions.py generates the integrated output diagram for the system dynamics model of the three specific institutions.
