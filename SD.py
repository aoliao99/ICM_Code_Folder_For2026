import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

class OccupationSDModel:
    def __init__(self, params, hist_data):
        """
        Initialize system dynamics model parameters
        params: Model parameters
        hist_data: Historical salary data dictionary
        """
        self.params = params
        self.hist = hist_data
        self.dt = params['dt']
        self.time_steps = int(params['simulation_years'] / self.dt)
        self.time = np.linspace(2024, 2024 + params['simulation_years'], self.time_steps)
        
        # --- Processing Historical Data ---
        # Calculate CAGR and Benchmark Premium Prior to Simulation
        self._process_historical_data()
        
        # --- Initialize state variables ---
        self.stocks = {
            'Students': np.zeros(self.time_steps),
            'Junior': np.zeros(self.time_steps),
            'Senior': np.zeros(self.time_steps),
            'Unemployed': np.zeros(self.time_steps)
        }
        
        self.stocks['Students'][0] = params['init_students']
        self.stocks['Junior'][0] = params['init_junior']
        self.stocks['Senior'][0] = params['init_senior']
        self.stocks['Unemployed'][0] = 0
        
        # --- Recorder  ---
        self.history = {
            'Market_Demand': [],
            'Gap': [],
            'Occ_Salary': [],       # Absolute Salary by Occupation
            'Social_Salary': [],    # Average Social Wage
            'Premium_Ratio': [],    # Salary Premium Multiple
            'AI_Pressure': [],
            'AI_Maturity': []
        }

    def _process_historical_data(self):
        """Calculate growth rates and benchmark values using historical data"""
        # A. Career Salary Trends
        occ_sal = np.array(self.hist['occupation_salary'])
        years_occ = np.arange(len(occ_sal)).reshape(-1, 1)
        # Calculate Career Salary Growth Rate
        model_occ = LinearRegression().fit(years_occ, np.log(occ_sal))
        self.occ_growth_rate = model_occ.coef_[0]
        self.base_occ_salary_2024 = occ_sal[-1]
        
        # B. Societal Salary Trends
        soc_sal = np.array(self.hist['social_salary'])
        years_soc = np.arange(len(soc_sal)).reshape(-1, 1)
        # Calculate the social wage growth rate
        model_soc = LinearRegression().fit(years_soc, np.log(soc_sal))
        self.soc_growth_rate = model_soc.coef_[0]
        self.base_soc_salary_2024 = soc_sal[-1]
        
        # C. Calculate the initial premium multiple for 2024 
        self.initial_premium = self.base_occ_salary_2024 / self.base_soc_salary_2024
        
        print(f"--- Data calculation completed ---")
        print(f"Career CAGR: {self.occ_growth_rate*100:.2f}% | Social AverageCAGR: {self.soc_growth_rate*100:.2f}%")
        print(f"Initial premium multiple: {self.initial_premium:.2f}x")

    def sigmoid(self, t, t0, speed):
        return 1 / (1 + np.exp(-speed * (t - t0)))

    def run(self):
        p = self.params
        
        # Precomputed coefficient C for Logistic growth
        # Assuming the initial total population approximates the natural demand at that time
        initial_emp = self.stocks['Junior'][0] + self.stocks['Senior'][0]
        C_logistic = (p['K'] / (initial_emp + 1e-9)) - 1 

        for t in range(self.time_steps - 1):
            curr_year = self.time[t]
            delta_year = curr_year - 2024
            
            S = self.stocks['Students'][t]
            J = self.stocks['Junior'][t]
            E = self.stocks['Senior'][t]
            # U renamed to “Cumulative Number of People Replaced by AI”
            U_cum = self.stocks['Unemployed'][t] 
            
            # --- 1. Dynamic Benchmark Salary ---
            curr_soc_salary = self.base_soc_salary_2024 * np.exp(self.soc_growth_rate * delta_year)
            natural_occ_salary = self.base_occ_salary_2024 * np.exp(self.occ_growth_rate * delta_year)
            
            # --- 2. Demand and AI  ---
            ai_maturity = self.sigmoid(curr_year, p['ai_boom_year'], p['ai_speed'])
            
            # A. Logistic Organic market growth
            natural_demand = p['K'] / (1 + C_logistic * np.exp(-p['r'] * delta_year))
            
            # B. Delivering the AI Impact 
            task_ratio = p.get('task_auto_ratio', 0.25)
            real_demand = natural_demand * (1 - p['alpha'] * ai_maturity * task_ratio)
            
            # --- Gap calculation ---
            total_supply = J + E
            gap = real_demand - total_supply
            gap_ratio = gap / (real_demand + 1e-9)
            
            # --- 3. Salary Feedback Chain ---
            salary_gap_effect = 1 + (gap_ratio * p['salary_sensitivity'])
            salary_gap_effect = np.clip(salary_gap_effect, 0.7, 1.5)
            actual_occ_salary = natural_occ_salary * salary_gap_effect
            
            current_premium = actual_occ_salary / curr_soc_salary
            attractiveness_factor = (current_premium / self.initial_premium) ** p['enrollment_elasticity']
            
            # AI Cost Pressure
            cost_push = 1 + (current_premium - self.initial_premium) * p['cost_push_factor']
            ai_adoption_pressure = ai_maturity * max(0.5, cost_push)
            
            # --- 4. Rate equation---
            
            # Enrollment Restrictions
            max_cap = p.get('max_school_capacity', p['base_enrollment'] * 3) # 默认给个上限
            enrollment = min(p['base_enrollment'] * attractiveness_factor, max_cap)
            
            grad_join = S / p['school_years']
            promotion = J * p['promotion_rate']
            
            j_exit_natural = J * p['natural_turnover']
            e_exit_natural = E * p['retirement_rate']
            
            j_ai_replace = J * p['alpha'] * ai_adoption_pressure * p['vuln_junior']
            e_ai_replace = E * p['alpha'] * ai_adoption_pressure * p['vuln_senior']
            
            # --- 5. Integral Calculus ---
            dS = enrollment - grad_join
            dJ = grad_join - promotion - j_exit_natural - j_ai_replace
            dE = promotion - e_exit_natural - e_ai_replace
            
            #  Cumulative AI Replacement Flow
            dU = j_ai_replace + e_ai_replace
            
            self.stocks['Students'][t+1] = S + dS * self.dt
            self.stocks['Junior'][t+1] = J + dJ * self.dt
            self.stocks['Senior'][t+1] = E + dE * self.dt
            self.stocks['Unemployed'][t+1] = U_cum + dU * self.dt
            
            
            # --- Record data ---
            self.history['Market_Demand'].append(real_demand)
            self.history['Gap'].append(gap_ratio)
            self.history['Occ_Salary'].append(actual_occ_salary)
            self.history['Social_Salary'].append(curr_soc_salary)
            self.history['Premium_Ratio'].append(current_premium)
            self.history['AI_Pressure'].append(ai_adoption_pressure)
            self.history['AI_Maturity'].append(ai_maturity)
            
        for k in self.history:
            self.history[k].append(self.history[k][-1])


    def plot_results(self):
        #AI Statement: This section utilizes Gemini 3 Pro to integrate four images into a single composite image.
        """ Comprehensive Analysis Chart """
        years = self.time
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # The relation between employment and market demand
        ax1 = axes[0, 0]
        ax1.plot(years, self.stocks['Junior'], label='Junior ', color='skyblue')
        ax1.plot(years, self.stocks['Senior'], label='Senior ', color='blue')
        ax1.plot(years, self.stocks['Unemployed'], label='Unemployed (replaced by AI)', color='red', linestyle='--')
        ax1.plot(years, self.history['Market_Demand'], label='Market Demand', color='green', linestyle=':')
        ax1.set_title('The relation between employment and market demand')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Premium_Ratio And AI Pressure
        ax2 = axes[0, 1]
        ax2.plot(years, self.history['Premium_Ratio'], label='Premium Ratio ', color='orange')
        ax2.plot(years, self.history['AI_Pressure'], label='AI Pressure ', color='purple')
        ax2.axhline(self.initial_premium, color='gray', linestyle='--', alpha=0.5, label='2024 Benchmark')
        ax2.set_title('The attraction of salary premium vs AI replacement pressure')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # The ratio of junior vs senior employees
        ax3 = axes[1, 0]
        total_emp = self.stocks['Junior'] + self.stocks['Senior']
        total_emp = np.maximum(total_emp, 1) 
        ax3.fill_between(years, 0, self.stocks['Junior']/total_emp, label='Junior Ratio', alpha=0.6)
        ax3.fill_between(years, self.stocks['Junior']/total_emp, 1, label='Senior Ratio', alpha=0.6)
        ax3.set_title('The ratio of junior vs senior employees')
        ax3.set_ylim(0, 1)
        ax3.legend()
        
        # accumulated number of AI-replaced unemployed
        ax4 = axes[1, 1]
        ax4.plot(years, self.stocks['Unemployed'], color='darkred', linewidth=2)
        ax4.set_title('accumulated number of AI-replaced unemployed')
        ax4.fill_between(years, 0, self.stocks['Unemployed'], color='red', alpha=0.1)
        ax4.grid(True)
        
        plt.tight_layout()
        plt.show()

    def plot_financial_analysis(self):
        """Detailed Salary Level Analysis Chart"""
        years = self.time
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Absolute Salary Forecast
        ax1 = axes[0]
        ax1.plot(years, self.history['Occ_Salary'], label='Occupation Salary', color='blue', linewidth=2)
        ax1.plot(years, self.history['Social_Salary'], label='Social Avg Salary', color='gray', linestyle='--')
        ax1.set_title('The absolute salary trend prediction ($/year)')
        ax1.set_ylabel('annual salary ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # High-salary trap analysis
        ax2 = axes[1]
        sc = ax2.scatter(self.history['Premium_Ratio'], self.history['AI_Pressure'], c=years, cmap='viridis')
        ax2.set_xlabel('Salary Premium')
        ax2.set_ylabel('AI Pressure')
        ax2.set_title('High-salary trap: Does high premium accelerate AI replacement?')
        plt.colorbar(sc, label='year')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()

# ==========================================
# Data Configuration Area 
# ==========================================

# 1. Historical Data Preparation
occ_history_data = [
    62857.6, 66019.2, 66435.2, 66768, 67579.2, 69929.6, 71884.8, 
74588.8, 78020.8, 83116.8, 86777.6, 86236.8, 104436.8, 107432, 114670.4
]
#Average Salary
social_history_data = [56310,58260,61900,65470,67920] 

# 2. Pack historical data
hist_data_package = {
    'occupation_salary': occ_history_data,
    'social_salary': social_history_data
}

# 3. Parameter Settings
programmer_params = {
    # Simulation Settings
    'dt': 0.1, 
    'simulation_years': 16,
    
    # Initial Stock Values
    'init_students': 13152, 
    'init_junior': 1800000,   
    'init_senior': 500000,

    # Growth Parameters
    'r':  0.0230,
    'K':  3811230,
    'alpha': 0.5224,

    # Education System Parameters
    'school_years': 4,
    'base_enrollment': 3056,
    'promotion_rate': 0.05,
    'natural_turnover': 0.10,
    'retirement_rate': 0.04,
    
    # AI Adoption Parameters
    'ai_boom_year': 2026,
    'ai_speed': 0.9,
    'vuln_junior': 0.20,
    'vuln_senior': 0.30,
    
    # Feedback Sensitivities
    'salary_sensitivity': 1.2,
    'enrollment_elasticity': 1.5,
    'cost_push_factor': 0.8
}

# 4. Run the Model
model = OccupationSDModel(programmer_params, hist_data_package)
model.run()

# 5. Plot the Results
model.plot_results()  
model.plot_financial_analysis() 
