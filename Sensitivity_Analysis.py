import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import copy

class OccupationSDModel:
    def __init__(self, params, hist_data):
        """
        Initialize system dynamics model parameters
        params: Model parameters
        hist_data: Historical salary data dictionary
        """
        # Use deep copying to prevent the original parameter dictionary from being modified during subsequent execution.
        self.params = copy.deepcopy(params)
        self.hist = hist_data
        self.dt = self.params['dt']
        self.time_steps = int(self.params['simulation_years'] / self.dt)
        self.time = np.linspace(2024, 2024 + self.params['simulation_years'], self.time_steps)

        # --- Process historical data ---
        self._process_historical_data()
        
        # --- Initialize state variables ---
        self.stocks = {
            'Students': np.zeros(self.time_steps),
            'Junior': np.zeros(self.time_steps),
            'Senior': np.zeros(self.time_steps),
            'Unemployed': np.zeros(self.time_steps)
        }
        
        self.stocks['Students'][0] = self.params['init_students']
        self.stocks['Junior'][0] = self.params['init_junior']
        self.stocks['Senior'][0] = self.params['init_senior']
        self.stocks['Unemployed'][0] = 0
        
        # --- Recorder ---
        self.history = {
            'Market_Demand': [], 'Gap': [], 'Occ_Salary': [], 'Social_Salary': [],
            'Premium_Ratio': [], 'AI_Pressure': [], 'AI_Maturity': []
        }

    def _process_historical_data(self):
        """Calculate growth rates and benchmark values using historical data"""
        occ_sal = np.array(self.hist['occupation_salary'])
        years_occ = np.arange(len(occ_sal)).reshape(-1, 1)
        model_occ = LinearRegression().fit(years_occ, np.log(occ_sal))
        self.occ_growth_rate = model_occ.coef_[0]
        self.base_occ_salary_2024 = occ_sal[-1]
        
        soc_sal = np.array(self.hist['social_salary'])
        years_soc = np.arange(len(soc_sal)).reshape(-1, 1)
        model_soc = LinearRegression().fit(years_soc, np.log(soc_sal))
        self.soc_growth_rate = model_soc.coef_[0]
        self.base_soc_salary_2024 = soc_sal[-1]
        
        self.initial_premium = self.base_occ_salary_2024 / self.base_soc_salary_2024

    def sigmoid(self, t, t0, speed):
        return 1 / (1 + np.exp(-speed * (t - t0)))

    def run(self):
        p = self.params
        initial_emp = self.stocks['Junior'][0] + self.stocks['Senior'][0]
        C_logistic = (p['K'] / (initial_emp + 1e-9)) - 1

        for t in range(self.time_steps - 1):
            curr_year = self.time[t]
            delta_year = curr_year - 2024
            
            S = self.stocks['Students'][t]
            J = self.stocks['Junior'][t]
            E = self.stocks['Senior'][t]
            U_cum = self.stocks['Unemployed'][t]
            
            curr_soc_salary = self.base_soc_salary_2024 * np.exp(self.soc_growth_rate * delta_year)
            natural_occ_salary = self.base_occ_salary_2024 * np.exp(self.occ_growth_rate * delta_year)
            
            ai_maturity = self.sigmoid(curr_year, p['ai_boom_year'], p['ai_speed'])
            natural_demand = p['K'] / (1 + C_logistic * np.exp(-p['r'] * delta_year))
            task_ratio = p.get('task_auto_ratio', 0.25)
            real_demand = natural_demand * (1 - p['alpha'] * ai_maturity * task_ratio)
            
            total_supply = J + E
            gap = real_demand - total_supply
            gap_ratio = gap / (real_demand + 1e-9)
            
            salary_gap_effect = np.clip(1 + (gap_ratio * p['salary_sensitivity']), 0.7, 1.5)
            actual_occ_salary = natural_occ_salary * salary_gap_effect
            
            current_premium = actual_occ_salary / curr_soc_salary
            attractiveness_factor = (current_premium / self.initial_premium) ** p['enrollment_elasticity']
            
            cost_push = 1 + (current_premium - self.initial_premium) * p['cost_push_factor']
            ai_adoption_pressure = ai_maturity * max(0.5, cost_push)
            
            max_cap = p.get('max_school_capacity', p['base_enrollment'] * 3)
            enrollment = min(p['base_enrollment'] * attractiveness_factor, max_cap)
            
            grad_join = S / p['school_years']
            promotion = J * p['promotion_rate']
            j_exit_natural = J * p['natural_turnover']
            e_exit_natural = E * p['retirement_rate']
            j_ai_replace = J * p['alpha'] * ai_adoption_pressure * p['vuln_junior']
            e_ai_replace = E * p['alpha'] * ai_adoption_pressure * p['vuln_senior']
            
            dS = enrollment - grad_join
            dJ = grad_join - promotion - j_exit_natural - j_ai_replace
            dE = promotion - e_exit_natural - e_ai_replace
            dU = j_ai_replace + e_ai_replace
            
            self.stocks['Students'][t+1] = S + dS * self.dt
            self.stocks['Junior'][t+1] = J + dJ * self.dt
            self.stocks['Senior'][t+1] = E + dE * self.dt
            self.stocks['Unemployed'][t+1] = U_cum + dU * self.dt
            
            self.history['Market_Demand'].append(real_demand)
            self.history['Gap'].append(gap_ratio)
            self.history['Occ_Salary'].append(actual_occ_salary)
            self.history['Social_Salary'].append(curr_soc_salary)
            self.history['Premium_Ratio'].append(current_premium)
            self.history['AI_Pressure'].append(ai_adoption_pressure)
            self.history['AI_Maturity'].append(ai_maturity)
            
        for k in self.history:
            if self.history[k]:
                self.history[k].append(self.history[k][-1])

    def plot_results(self):
        """ Comprehensive Analysis Chart """
        years = self.time
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        ax1 = axes[0, 0]
        ax1.plot(years, self.stocks['Junior'], label='Junior', color='skyblue')
        ax1.plot(years, self.stocks['Senior'], label='Senior', color='blue')
        ax1.plot(years, self.stocks['Unemployed'], label='Unemployed (AI)', color='red', linestyle='--')
        ax1.plot(years, self.history['Market_Demand'], label='Market Demand', color='green', linestyle=':')
        ax1.set_title('Employment vs Market Demand')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax2 = axes[0, 1]
        ax2.plot(years, self.history['Premium_Ratio'], label='Premium Ratio', color='orange')
        ax2.plot(years, self.history['AI_Pressure'], label='AI Pressure', color='purple')
        ax2.axhline(self.initial_premium, color='gray', linestyle='--', alpha=0.5, label=f'2024 Benchmark ({self.initial_premium:.2f}x)')
        ax2.set_title('Salary Premium vs AI Replacement Pressure')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax3 = axes[1, 0]
        total_emp = np.maximum(self.stocks['Junior'] + self.stocks['Senior'], 1)
        ax3.fill_between(years, 0, self.stocks['Junior']/total_emp, label='Junior Ratio', alpha=0.6)
        ax3.fill_between(years, self.stocks['Junior']/total_emp, 1, label='Senior Ratio', alpha=0.6)
        ax3.set_title('Ratio of Junior vs Senior Employees')
        ax3.set_ylim(0, 1)
        ax3.legend()
        ax4 = axes[1, 1]
        ax4.plot(years, self.stocks['Unemployed'], color='darkred', linewidth=2)
        ax4.set_title('Accumulated AI-Replaced Unemployed')
        ax4.fill_between(years, 0, self.stocks['Unemployed'], color='red', alpha=0.1)
        ax4.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_financial_analysis(self):
        """Detailed Salary Level Analysis Chart"""
        years = self.time
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        ax1 = axes[0]
        ax1.plot(years, self.history['Occ_Salary'], label='Occupation Salary', color='blue', linewidth=2)
        ax1.plot(years, self.history['Social_Salary'], label='Social Avg Salary', color='gray', linestyle='--')
        ax1.set_title('Absolute Salary Trend Prediction ($/year)')
        ax1.set_ylabel('Annual Salary ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax2 = axes[1]
        sc = ax2.scatter(self.history['Premium_Ratio'], self.history['AI_Pressure'], c=years, cmap='viridis')
        ax2.set_xlabel('Salary Premium Ratio')
        ax2.set_ylabel('AI Adoption Pressure')
        ax2.set_title('High-Salary Trap: Premium vs. AI Pressure')
        plt.colorbar(sc, label='Year')
        ax2.grid(True)
        plt.tight_layout()
        plt.show()

# ==========================================
# SENSITIVITY ANALYSIS FUNCTIONS 
# ==========================================

def get_output_metrics(model):
    """
    Extract specified output metrics from an already running model instance.
    """
    # Avoid calling min/max on an empty history.
    if not model.history['Premium_Ratio']:
        premium_min = np.nan
    else:
        premium_min = np.min(model.history['Premium_Ratio'])

    # Avoid division by zero at the last moment
    total_emp_end = model.stocks['Junior'][-1] + model.stocks['Senior'][-1]
    if total_emp_end == 0:
        senior_ratio_end = np.nan
    else:
        senior_ratio_end = model.stocks['Senior'][-1] / total_emp_end

    metrics = {
        'Unemployed_End': model.stocks['Unemployed'][-1],
        'Junior_Peak_Year': model.time[np.argmax(model.stocks['Junior'])],
        'Premium_Min': premium_min,
        'Senior_Ratio_End': senior_ratio_end
    }
    return metrics

def run_sensitivity_analysis(base_params, hist_data, params_to_test, change_percent=0.2):
    """
    Perform a complete sensitivity analysis process and print formatted result tables.
    """
    all_results = []

    # 1. Run benchmark simulation
    print("--- Running Baseline Simulation for Sensitivity Analysis ---")
    base_model = OccupationSDModel(base_params, hist_data)
    base_model.run()
    baseline_metrics = get_output_metrics(base_model)
    all_results.append({'Parameter': 'Baseline', 'Scenario': 'Baseline', **baseline_metrics})
    print("Baseline Metrics Calculated.")

    # 2. Test each parameter in a loop
    for param_name in params_to_test:
        base_value = base_params[param_name]
        
        # --- Low Scenario ---
        low_params = copy.deepcopy(base_params)
        low_value = base_value * (1 - change_percent)
        low_params[param_name] = low_value
        print(f"\n--- Testing {param_name} (Low: {low_value:.4f}) ---")
        low_model = OccupationSDModel(low_params, hist_data)
        low_model.run()
        low_metrics = get_output_metrics(low_model)
        all_results.append({'Parameter': param_name, 'Scenario': f'Low (-{change_percent*100}%)', **low_metrics})

        # --- High Scenario ---
        high_params = copy.deepcopy(base_params)
        high_value = base_value * (1 + change_percent)
        high_params[param_name] = high_value
        print(f"--- Testing {param_name} (High: {high_value:.4f}) ---")
        high_model = OccupationSDModel(high_params, hist_data)
        high_model.run()
        high_metrics = get_output_metrics(high_model)
        all_results.append({'Parameter': param_name, 'Scenario': f'High (+{change_percent*100}%)', **high_metrics})

    # 3. Format and print the results
    print("\n\n" + "="*80)
    print("SENSITIVITY ANALYSIS RESULTS".center(80))
    print("="*80)
    
    # Generating visually appealing tables using pandas DataFrames
    results_df = pd.DataFrame(all_results)
    return results_df

# ==========================================
# NEW & IMPROVED TORNADO PLOT FUNCTION
# ==========================================
def plot_tornado(results_df, output_metric_to_plot):
    #AI Statement: This section utilized Gemini3pro to refine the tornado diagram code and correct errors within this portion of the code.
    """
    Plot a tornado chart based on the resultsDataFrame from the sensitivity analysis.
    :param results_df: The DataFrame generated by run_sensitivity_analysis.
    :param output_metric_to_plot: The column name of the output metric to plot (e.g., ‘Unemployed_End’).
    """
    # 1. Data Preparation
    baseline_value = results_df.loc[results_df['Parameter'] == 'Baseline', output_metric_to_plot].iloc[0]
    params_df = results_df[results_df['Parameter'] != 'Baseline'].copy()
    pivoted = params_df.pivot(index='Parameter', columns='Scenario', values=output_metric_to_plot)
    
    low_col = [col for col in pivoted.columns if 'Low' in col][0]
    high_col = [col for col in pivoted.columns if 'High' in col][0]

    pivoted['range'] = abs(pivoted[high_col] - pivoted[low_col])
    sorted_df = pivoted.sort_values(by='range', ascending=False)
    
    # 2. Drawing
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 8))

    param_names = sorted_df.index
    y_pos = np.arange(len(param_names))

    high_values = sorted_df[high_col]
    low_values = sorted_df[low_col]

    ax.barh(y_pos, high_values - baseline_value, left=baseline_value, color='lightcoral', label='High Scenario (+20%)')
    ax.barh(y_pos, low_values - baseline_value, left=baseline_value, color='cornflowerblue', label='Low Scenario (-20%)')

    ax.axvline(x=baseline_value, color='red', linestyle='--', label=f'Baseline: {baseline_value:.2f}')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(param_names)
    ax.invert_yaxis()
    ax.set_xlabel(f'Value of {output_metric_to_plot}')
    ax.set_title(f'Tornado Plot for: {output_metric_to_plot}')
    
    # Intelligently determine which formatting tool to use
    if 'Year' in output_metric_to_plot:
        # For years, use a simple integer format.
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0f}'))
    else:
        # Otherwise, use the format with a thousand-separator.
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.4f}'))

    ax.legend()
    plt.tight_layout()
    plt.show()


# ==========================================
# DATA & PARAMETERS
# ==========================================

occ_history_data = [
    62857.6, 66019.2, 66435.2, 66768, 67579.2, 69929.6, 71884.8, 
 74588.8, 78020.8, 83116.8, 86777.6, 86236.8, 104436.8, 107432, 114670.4
]
social_history_data = [56310, 58260, 61900, 65470, 67920]
hist_data_package = {
    'occupation_salary': occ_history_data,
    'social_salary': social_history_data
}

programmer_params = {
    'dt': 0.1, 'simulation_years': 16,
    'init_students': 10000, 'init_junior': 35000, 'init_senior': 8000,
    'r':-0.0189, 'K': 46310, 'alpha': 0.3411,
    'school_years':4, 'base_enrollment': 2000, 'promotion_rate': 0.05,
    'natural_turnover': 0.10, 'retirement_rate': 0.04,
    'ai_boom_year': 2026, 'ai_speed': 0.9,
    'vuln_junior': 0.20, 'vuln_senior': 0.30,
    'salary_sensitivity': 1.2, 'enrollment_elasticity': 1.5, 'cost_push_factor': 0.8
}

# ==========================================
# EXECUTION
# ==========================================

# 1. Run the original model and plot the results. 
print("--- Running Base Model & Plotting ---")
base_model_main = OccupationSDModel(programmer_params, hist_data_package)
base_model_main._process_historical_data() # Print initial stats
base_model_main.run()
base_model_main.plot_results()
base_model_main.plot_financial_analysis()

# 1. Operational Sensitivity Analysis
params_to_test = [
    'alpha', 'r', 'K', 'ai_speed', 'salary_sensitivity', 
    'enrollment_elasticity', 'cost_push_factor'
]
results_df = run_sensitivity_analysis(
    base_params=programmer_params,
    hist_data=hist_data_package,
    params_to_test=params_to_test,
    change_percent=0.2
)
# 2. Check and print the received DataFrame
print("\n" + "="*80)
print("SENSITIVITY ANALYSIS RESULTS".center(80))
print("="*80)
# Set pandas print format
pd.set_option('display.width', 120)
pd.set_option('display.max_columns', 10)
pd.set_option('display.float_format', '{:.4f}'.format)
print(results_df) 

# 3. Plotting a tornado diagram
if results_df is not None:
    print("\n" + "#"*80)
    print("PLOTTING TORNADO DIAGRAM".center(80))
    print("#"*80)
    plot_tornado(results_df, 'Unemployed_End')
    plot_tornado(results_df, 'Premium_Min')
else:
    print("Error: Sensitivity analysis did not return a valid DataFrame. Cannot plot.")
