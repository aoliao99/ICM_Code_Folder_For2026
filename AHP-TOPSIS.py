import pandas as pd
import numpy as np
import io

# ==========================================
# 1. Data loading 
# ==========================================
csv_content = """Occupation,Automation,Pace,Exactness,Email,Programming,ProblemSolving,Social,Originality,Manual,Deductive
Programmer,32,3,81,100,94,69,53,50,25,69
Chef,20,36,78,18,0,35,50,40,55,59
Singer,2,19,90,63,6,35,53,60,53,47
Counselor,12,0,64,91,16,69,85,60,0,75
Data_Entry,46,21,97,83,19,47,41,25,25,47
"""

df = pd.read_csv(io.StringIO(csv_content), index_col="Occupation")

print("--- 1.  Confirm the data read ---")
print(df)
print("\n")

# ==========================================
# 2. Core Configuration: Indicator Direction
# ==========================================
# 1 = Benefit (Positive: The higher, the safer)
# -1 = Cost (Negative: The higher the value, the greater the risk/likelihood of being replaced.)
direction_map = {
    "Automation": -1,     
    "Pace": -1,           
    "Exactness": -1,      
    "Email": -1,          
    "Programming": -1,    
    "ProblemSolving": 1,  
    "Social": 1,          
    "Originality": 1,    
    "Manual": 1,          
    "Deductive": 1        
}

indicators = list(df.columns)
directions = np.array([direction_map[col] for col in indicators])

# ==========================================
# 3. AHP Weight Calculation 
# ==========================================
# Construct a weight vector.
# Logic: Originality and Social are critical for survival in the AI era; assign high scores.
# Programming and Automation are risk factors; assign medium-to-high scores.
# Assign low scores to all others.

n = len(indicators)
ahp_df = pd.DataFrame(np.ones((n, n)), index=indicators, columns=indicators)

def set_rel(row, col, val):
    ahp_df.loc[row, col] = val
    ahp_df.loc[col, row] = 1/val

# --- Construct a comparison matrix ---
# 1. Establish Originality as the highest priority
for col in indicators:
    if col != "Originality":
        set_rel("Originality", col, 4)

# 2. Establishing the Status of Social
set_rel("Social", "Automation", 3)
set_rel("Social", "Email", 4)
set_rel("Social", "Pace", 4)
set_rel("Social", "Programming", 2)

# 3. Establishing the Importance of Programming (as a Risk Indicator)
set_rel("Programming", "Email", 3)
set_rel("Programming", "Pace", 3)
set_rel("Programming", "Exactness", 2)

# 4. Establishing the Manual (Agile) Position (For Chefs/Singers)
set_rel("Manual", "Email", 3)

# Calculate weights
mat = ahp_df.values
w = np.mean(mat / np.sum(mat, axis=0), axis=1)
weights = pd.Series(w, index=indicators)

print("--- 2. AHP Weight (Check if Originality is the highest) ---")
print(weights.sort_values(ascending=False))
print("\n")

# ==========================================
# AHP Consistency Verification Module
# ==========================================
# 1.  Calculate the maximum eigenvalue  lambda_max
Aw = np.dot(mat, w)
lambda_max = np.mean(Aw / w)

# 2. Calculate CI (Consistency Index)
CI = (lambda_max - n) / (n - 1)

# 3. Search RI (Random Index)
RI_dict = {
    1: 0, 2: 0, 3: 0.58, 4: 0.90, 5: 1.12, 
    6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49, 11: 1.51
}
RI = RI_dict.get(n, 1.49) 

# 4. Calculate CR (Consistency Ratio)
if RI == 0:
    CR = 0
else:
    CR = CI / RI

print(f"--- [AHP Consistency Verification] ---")
print(f"n (Indicator Number) = {n}")
print(f"Lambda_max = {lambda_max:.4f}")
print(f"CI = {CI:.4f}")
print(f"RI = {RI}")
print(f"CR = {CR:.4f}")

if CR < 0.1:
    print(" Consistency check passed(CR < 0.1)，The weight allocation logic is reasonable.")
else:
    print("Consistency check failed (CR >= 0.1)。")
    print("Recommendation: Re-adjust the values in `set_rel` to avoid logical inconsistencies where A > B, B > C, but C > A.")
print("\n")


# ==========================================
# 4. TOPSIS Calculation
# ==========================================
raw = df.values.astype(float)
processed = raw.copy()

# Step 1: Normalization (Cost -> Benefit)
# For negative indicators (such as Automation), the higher the value, the greater the risk.
# We use Max - x to convert it into “safety.”
# Data_Entry Automation=46 (higher), the converted score will be lower.
for i, col in enumerate(indicators):
    if directions[i] == -1:
        processed[:, i] = np.max(raw[:, i]) - raw[:, i]

# Step 2: Normalization
denom = np.sqrt(np.sum(processed**2, axis=0))
denom[denom == 0] = 1
norm = processed / denom

# Step 3: weighted
weighted = norm * weights.values

# Step 4: Distance Calculation
ideal_best = np.max(weighted, axis=0)
ideal_worst = np.min(weighted, axis=0)

d_plus = np.sqrt(np.sum((weighted - ideal_best)**2, axis=1))
d_minus = np.sqrt(np.sum((weighted - ideal_worst)**2, axis=1))

# Step 5: Scoring
# Resilience (Survival Score) = Distance from Worst Solution / Total Distance
# Higher Score -> Greater Safety
resilience_score = d_minus / (d_plus + d_minus)

# ==========================================
# 5. Result Output
# ==========================================
result = df.copy()
result["AI_Resilience "] = resilience_score
result["AI_Risk "] = 1 - resilience_score 

# Ranked from highest to lowest risk
result = result.sort_values("AI_Risk ", ascending=False)

print("--- 3. Final Ranking ---")
print(result[["AI_Risk ", "AI_Resilience "]])

# Extract parameters
print("\n--- [Parameter Extraction] Please fill the following Alpha values into your SD model ---")
for idx, row in result.iterrows():
    print(f"Alpha ({idx}): {row['AI_Risk ']:.4f}")

# Simple logical verification  (AI Statement:This section was generated using Gemini3pro.)
print("\n--- Simple logical verification ---")
if result.index[0] == "Data_Entry":
    print("Verification passed: Data_Entry poses the highest risk. ")
elif result.index[0] == "Programmer":
    print("Note: Programmer carries the highest risk. ")
else:
    print(f"The result is unreasonable: {result.index[0]} poses the highest risk. Please verify the data.")

# 保存
result.to_csv("Final_Risk_Assessment.csv")
print("\n Results have been saved to Final_Risk_Assessment.csv")
