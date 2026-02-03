import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import matplotlib.ticker as ticker

class OccupationSDModel:
    def __init__(self, params, hist_data):
        """
        Initialize system dynamics model parameters
        """
        self.params = params
        self.hist = hist_data
        self.dt = params['dt']
        self.time_steps = int(params['simulation_years'] / self.dt)
        self.time = np.linspace(2024, 2024 + params['simulation_years'], self.time_steps)
        
        # --- Process historical data ---
        self._process_historical_data()
        
        # --- Initialize state variables ---
        self.stocks = {
            'Students': np.zeros(self.time_steps),
            'Junior': np.zeros(self.time_steps),
            'Senior': np.zeros(self.time_steps),
            'Unemployed': np.zeros(self.time_steps)
        }
        
        # Initial value
        self.stocks['Students'][0] = params['init_students']
        self.stocks['Junior'][0] = params['init_junior']
        self.stocks['Senior'][0] = params['init_senior']
        self.stocks['Unemployed'][0] = 0
        
        # --- Recorder ---
        self.history = {
            'Market_Demand': [],
            'Gap': [],
            'Occ_Salary': [],
            'Social_Salary': [],
            'Premium_Ratio': [],
            'ROI_Value': [],       
            'AI_Pressure': [],
            'AI_Maturity': []
        }

    def _process_historical_data(self):
        """Calculate growth rates and benchmark values using historical data"""
        # A. Career Salary Trends
        occ_sal = np.array(self.hist['occupation_salary'])
        years_occ = np.arange(len(occ_sal)).reshape(-1, 1)
        model_occ = LinearRegression().fit(years_occ, np.log(occ_sal))
        self.occ_growth_rate = model_occ.coef_[0]
        self.base_occ_salary_2024 = occ_sal[-1]
        
        # B. Societal Salary Trends
        soc_sal = np.array(self.hist['social_salary'])
        years_soc = np.arange(len(soc_sal)).reshape(-1, 1)
        model_soc = LinearRegression().fit(years_soc, np.log(soc_sal))
        self.soc_growth_rate = model_soc.coef_[0]
        self.base_soc_salary_2024 = soc_sal[-1]
        
        # C. Calculating the Initial ROI Benchmark for 2024 
        # Initial Annualized ROI = (Career Annual Salary - Annual Tuition Amortization) / Average Annual Salary
        # Note: For simplicity, we assume tuition pressure primarily impacts the first few years after graduation and use a smoothed annualized deduction.
        annual_tuition_cost = self.params['tuition_fee'] 
        
        # Net Premium
        net_income = self.base_occ_salary_2024 - (annual_tuition_cost / 10) 
        self.initial_roi_index = net_income / self.base_soc_salary_2024
        
        print(f"--- Data calculation completed ---")
        print(f"Initial ROI Index: {self.initial_roi_index:.2f} (Considered Tuition: ${annual_tuition_cost})")

    def sigmoid(self, t, t0, speed):
        return 1 / (1 + np.exp(-speed * (t - t0)))

    def run(self):
        p = self.params
        
        # Logistic C
        initial_emp = self.stocks['Junior'][0] + self.stocks['Senior'][0]
        C_logistic = (p['K'] / (initial_emp + 1e-9)) - 1 

        for t in range(self.time_steps - 1):
            curr_year = self.time[t]
            delta_year = curr_year - 2024
            
            S = self.stocks['Students'][t]
            J = self.stocks['Junior'][t]
            E = self.stocks['Senior'][t]
            U_cum = self.stocks['Unemployed'][t] 
            
            # --- 1. Dynamic Benchmark Salary ---
            curr_soc_salary = self.base_soc_salary_2024 * np.exp(self.soc_growth_rate * delta_year)
            natural_occ_salary = self.base_occ_salary_2024 * np.exp(self.occ_growth_rate * delta_year)
            
            # --- 2. Demand and AI ---
            ai_maturity = self.sigmoid(curr_year, p['ai_boom_year'], p['ai_speed'])
            natural_demand = p['K'] / (1 + C_logistic * np.exp(-p['r'] * delta_year))
            task_ratio = p.get('task_auto_ratio', 0.25)
            real_demand = natural_demand * (1 - p['alpha'] * ai_maturity * task_ratio)
            
            # Supply-demand gap
            total_supply = J + E
            gap = real_demand - total_supply
            gap_ratio = gap / (real_demand + 1e-9)
            
            # --- 3. Salary Feedback Chain ---
            salary_gap_effect = 1 + (gap_ratio * p['salary_sensitivity'])
            salary_gap_effect = np.clip(salary_gap_effect, 0.7, 1.5)
            actual_occ_salary = natural_occ_salary * salary_gap_effect
            
            # ROI-Driven Attractiveness Model
            # Here, ROI does not refer strictly to financial ROI, but rather a “relative attractiveness index”
            # Brand Premium: Brand premium (Top-tier universities >1, average universities = 1)
            # Tuition Inflation: Assumes tuition increases by 3% annually
            current_tuition = p['tuition_fee'] * (1.03 ** delta_year)
            
            # Expected Net Annual Income = Actual Salary - (Current Tuition / 10-Year Amortization)
            net_income_expected = actual_occ_salary - (current_tuition / 10)
            
            # Current ROI Index
            current_roi_index = net_income_expected / curr_soc_salary
            
            # Attractiveness = (Current ROI / Initial ROI) * Brand Bonus
            # If a prestigious institution (Brand > 1), less sensitive to ROI decline
            roi_ratio = current_roi_index / (self.initial_roi_index + 1e-5)
            attractiveness_factor = (roi_ratio ** p['enrollment_elasticity']) * p['brand_premium']
            
            #AI Cost Pressure
            current_pure_premium = actual_occ_salary / curr_soc_salary
            cost_push = 1 + (current_pure_premium - 1.2) * p['cost_push_factor'] # 假设1.2是正常倍数
            ai_adoption_pressure = ai_maturity * max(0.5, cost_push)
            
            # --- 4. Rate equation ---
            # Admission Restrictions
            max_cap = p.get('max_school_capacity', p['base_enrollment'] * 3)
            enrollment = min(p['base_enrollment'] * attractiveness_factor, max_cap)
            
            grad_join = S / p['school_years']
            promotion = J * p['promotion_rate']
            
            j_exit_natural = J * p['natural_turnover']
            e_exit_natural = E * p['retirement_rate']
            
            j_ai_replace = J * p['alpha'] * ai_adoption_pressure * p['vuln_junior']
            e_ai_replace = E * p['alpha'] * ai_adoption_pressure * p['vuln_senior']
            
            # --- 5. Integral Calculation ---
            dS = enrollment - grad_join
            dJ = grad_join - promotion - j_exit_natural - j_ai_replace
            dE = promotion - e_exit_natural - e_ai_replace
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
            self.history['Premium_Ratio'].append(current_pure_premium)
            self.history['ROI_Value'].append(current_roi_index) ### NEW
            self.history['AI_Pressure'].append(ai_adoption_pressure)
            self.history['AI_Maturity'].append(ai_maturity)
            
        for k in self.history:
            self.history[k].append(self.history[k][-1])

    def plot_results(self):
        #AI Statement:For chart generation, this section utilizes Gemini3pro to adjust line widths and image positioning in the generated diagrams, employing AI to integrate four images into a single large display.
        """ Comprehensive Analysis Chart (Including ROI) """
        years = self.time
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Figure 1: Enrollment & Stock
        ax1 = axes[0, 0]
        ax1.plot(years, self.stocks['Junior'], label='Junior Staff', color='skyblue')
        ax1.plot(years, self.stocks['Senior'], label='Senior Staff', color='blue')
        ax1.plot(years, self.stocks['Unemployed'], label='Displaced by AI', color='red', linestyle='--')
        ax1.set_title(f'Enrollment & Stock: {self.params["school_name"]}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Figure 2: Education ROI Trend
        ax2 = axes[0, 1]
        ax2.plot(years, self.history['ROI_Value'], label='Net ROI Index', color='green', linewidth=2)
        ax2.axhline(self.initial_roi_index, color='gray', linestyle='--', label='Initial ROI Benchmark')
        ax2.set_title('Education ROI Trend (Net Income / Social Avg)')
        ax2.set_ylabel('ROI Index')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Figure 3: Enrollment Demand (Reflecting Attractiveness)
        ax3 = axes[1, 0]
        est_enrollment = self.stocks['Students'] / self.params['school_years']
        ax3.plot(years, est_enrollment, color='purple', label='Est. Annual Enrollment')
        ax3.axhline(self.params['base_enrollment'], color='gray', linestyle=':', label='Base Capacity')
        ax3.set_title('School Enrollment Demand')
        ax3.legend()
        ax3.grid(True)
        
        # Figure 4: Salary vs AI Pressure
        ax4 = axes[1, 1]
        ax4.plot(years, self.history['Occ_Salary'], color='blue', label='Actual Salary')
        ax4.set_ylabel('Salary ($)', color='blue')
        ax2_twin = ax4.twinx()
        ax2_twin.plot(years, self.history['AI_Pressure'], color='red', linestyle='--', label='AI Pressure')
        ax2_twin.set_ylabel('AI Adoption Pressure', color='red')
        ax4.set_title('Salary vs AI Pressure')
        ax4.grid(True)
        
        plt.tight_layout()
        plt.show()

# ==========================================
# Data Configuration for MIT (Programmer)
# ==========================================

# 1. Historical Data 
occ_history_data = [62857, 66019, 66435, 66768, 67579, 69929, 71884, 74588, 78020, 83116, 86777, 86236, 104436, 107432, 114670]
social_history_data = [56310, 58260, 61900, 65470, 67920]
hist_data_package = {'occupation_salary': occ_history_data, 'social_salary': social_history_data}

# 2. MIT Specific parameters
mit_params = {
    'school_name': 'MIT (Computer Science)',
    
    'tuition_fee': 54518,       # 每年学费 ($)
    'brand_premium': 1.2,       # Brand Premium
    
    # Simulation Settings
    'dt': 0.1, 
    'simulation_years': 16,
    
    # Initial state
    'init_students': 13152,      
    'init_junior': 50000,     
    'init_senior': 20000,
    
    'r': 0.023,
    'K': 3811230,
    'alpha': 0.5224,
    
    'school_years': 4,
    'base_enrollment': 3056,   
    'promotion_rate': 0.08,    
    'natural_turnover': 0.08,
    'retirement_rate': 0.04,
    
    'ai_boom_year': 2026,
    'ai_speed': 0.9,
    'vuln_junior': 0.15,   
    'vuln_senior': 0.25,
    
    'salary_sensitivity': 1.2,
    'enrollment_elasticity': 1.5,
    'cost_push_factor': 0.8
}

# 3. Run Model
model = OccupationSDModel(mit_params, hist_data_package)
model.run()
model.plot_results()
