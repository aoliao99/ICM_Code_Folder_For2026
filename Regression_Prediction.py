import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. Data Entry (Complete 15-Year Data: 2010–2024)
years = np.arange(2010, 2025).reshape(-1, 1)

programmer_data = np.array([
    349980, 340470, 1901420, 1943450, 1992140, 2042300, 2072320, 2057330,
    2092430, 2157130, 2189480, 2236140, 2418260, 2540820, 2497660
])

chef_data = np.array([
   2096510, 2126170, 7136660, 7395770, 7650750, 7845510, 8005680, 8089140,
   8153730, 8503920, 7447470, 8407940, 8916540, 9053720, 9301660
])

singer_data = np.array([
    43350, 42530, 42100, 39260, 38900, 37090, 40110, 40170,
    41680, 41130, 34770, 24080, 31750, 35520, 38350
])

# 2. Define functions: Polynomials for plotting, statistical methods for computation
def analyze_and_plot_final(name, y_data, color):
    plt.figure(figsize=(10, 6))
    
    # -------------------------------------------------------
    # A: Visuals - Use a third-degree polynomial to fit the trend line
    # -------------------------------------------------------
    plt.scatter(years, y_data, color=color, s=80, edgecolors='k', zorder=5, label='Historical Data')
    
    years_smooth = np.linspace(2010, 2030, 200).reshape(-1, 1)
    poly = PolynomialFeatures(degree=3)
    X_poly = poly.fit_transform(years)
    
    model_viz = LinearRegression()
    model_viz.fit(X_poly, y_data)
    y_smooth_pred = model_viz.predict(poly.transform(years_smooth))
    
    r2_viz = r2_score(y_data, model_viz.predict(X_poly))
    
    plt.plot(years_smooth, y_smooth_pred, color='red', linewidth=2, linestyle='--', 
             label=f'Trend Fit (Poly-3, R²={r2_viz:.3f})')
    
    # Mark the Future Point
    for fy in [2025, 2030]:
        pred_val = model_viz.predict(poly.transform([[fy]]))[0]
        plt.plot(fy, pred_val, 'r*', markersize=15)
        plt.text(fy, pred_val, f'{int(pred_val):,}', ha='center', va='bottom', fontsize=15)
    #AI Statement: This section utilized Gemini 3 Pro to reasonably adjust the size of text annotations within the image.
    plt.title(f'Baseline Scenario: {name} (Data & Trend)', fontsize=20)
    plt.xlabel('Year', fontsize=20)
    plt.ylabel('Employment', fontsize=18)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper left')
    plt.gca().get_yaxis().set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    plt.xticks(fontsize=18) 
    plt.yticks(fontsize=18)
    # -------------------------------------------------------
    # B: SD Parameter Calculation (Model Inputs) 
    # -------------------------------------------------------
    # Extract the data index for 2012-2024 
    y_stable = y_data[2:] 
    # The corresponding X-axis (with 2012 set as t=0 for convenience in calculating r)
    t_stable = np.arange(len(y_stable))
    
    # --- Calculate r (natural growth rate)  ---
    if np.min(y_stable) > 0:
        log_y = np.log(y_stable)
        slope, intercept = np.polyfit(t_stable, log_y, 1)
        r_calc = slope
        
        # Supplement: Calculate the R² for the log regression to determine whether the correlation coefficient r is reliable.
        y_log_pred = slope * t_stable + intercept
        r2_log = r2_score(log_y, y_log_pred)
        
        confidence_msg = f"(Log-Linear R²={r2_log:.2f})"
        if r2_log < 0.5:
            confidence_msg += " [Warning: The trend in the index is not pronounced; the R-value is for reference only.]"
    else:
        r_calc = 0
        confidence_msg = "[Data contains zeros, making it impossible to perform log regression.]"

    # --- Calculate K (Carrying Capacity) ---
    # Logic: Detect the current growth trend
    if r_calc > 0.005: 
        status = "Growth Phase (Unsaturated)"
        K_calc = np.max(y_stable) * 1.5
        K_note = "Set to 1.5x Max (Market not saturated)"
    else:
        status = "Mature/Decline Phase"
        K_calc = np.max(y_stable) * 1.1
        K_note = "Set to 1.1x Peak"

    print("=" * 50)
    print(f"Occupation: {name}")
    print(f"--- SD Model Input Parameters (Based on 2012-2024 Stable Trend, Logistic Fitting) ---")
    print(f"1. Natural Growth Rate (r) : {r_calc:.4f} (i.e., {r_calc:.2%})")
    print(f"   (Interpretation: Annual expansion rate under no interference)")
    print(f"2. Carrying Capacity (K) : {int(K_calc):,}")
    print(f"   (Calculation Logic: {K_note})")
    print(f"--- Status Determination: {status}")
    print("=" * 50)

analyze_and_plot_final('Programmer', programmer_data, 'blue')
analyze_and_plot_final('Chef', chef_data, 'orange')
analyze_and_plot_final('Singer', singer_data, 'green')

plt.show()
