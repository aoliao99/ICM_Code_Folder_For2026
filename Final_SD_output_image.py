import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# ==========================================
# PART 1: SD Model Class Definition
# ==========================================
class OccupationSDModel:
    def __init__(self, params, hist_data):
        self.params = params
        self.hist = hist_data
        self.dt = params['dt']
        self.time_steps = int(params['simulation_years'] / self.dt)
        self.time = np.linspace(2024, 2024 + params['simulation_years'], self.time_steps)
        self._process_historical_data()
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

    def _process_historical_data(self):
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
            S, J, E, U_cum = self.stocks['Students'][t], self.stocks['Junior'][t], self.stocks['Senior'][t], self.stocks['Unemployed'][t]
            
            curr_soc_salary = self.base_soc_salary_2024 * np.exp(self.soc_growth_rate * delta_year)
            natural_occ_salary = self.base_occ_salary_2024 * np.exp(self.occ_growth_rate * delta_year)
            
            ai_maturity = self.sigmoid(curr_year, p['ai_boom_year'], p['ai_speed'])
            natural_demand = p['K'] / (1 + C_logistic * np.exp(-p['r'] * delta_year))
            task_ratio = p.get('task_auto_ratio', 0.25)
            real_demand = natural_demand * (1 - p['alpha'] * ai_maturity * task_ratio)
            
            gap = real_demand - (J + E)
            gap_ratio = gap / (real_demand + 1e-9)
            salary_gap_effect = np.clip(1 + (gap_ratio * p['salary_sensitivity']), 0.7, 1.5)
            actual_occ_salary = natural_occ_salary * salary_gap_effect
            current_premium = actual_occ_salary / curr_soc_salary
            
            # Attractiveness and Enrollment
            attractiveness_factor = (current_premium / self.initial_premium) ** p['enrollment_elasticity']
            
            # AI Cost-driven pressure
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
            
        return self.stocks

# ==========================================
# PART 2: Parameter Configuration and Batch Execution(AI Statement:This section utilizes Gemini3pro to optimize the generated diagram.)
# ==========================================
social_history_data = [56310, 58260, 61900, 65470, 67920]

configs = [
    {
        'name': 'Programmer',
        'color': '#1f77b4', 
        'salary_data': [61650, 62235, 63980, 65497.5, 66965, 68642.5, 70115, 72217.5, 73575, 76220, 78335, 80307.5, 85627.5, 89775, 91167.5],
        'params': {'dt': 0.1, 'simulation_years': 16, 'init_students': 600000, 'init_junior': 1800000, 'init_senior': 500000, 'r': 0.0230, 'K': 3811230, 'alpha': 0.5224, 'school_years': 4, 'base_enrollment': 180000, 'promotion_rate': 0.15, 'natural_turnover': 0.15, 'retirement_rate': 0.05, 'ai_boom_year': 2028, 'ai_speed': 0.8, 'vuln_junior': 0.25, 'vuln_senior': 0.05, 'salary_sensitivity': 1.5, 'enrollment_elasticity': 2.0, 'cost_push_factor': 1.2}
    },
    {
        'name': 'Chef',
        'color': '#ff7f0e', 
        'salary_data': [26784, 27187, 25725, 25872, 26031, 27746, 29186, 30286, 31122, 32922, 33855, 34362, 36772, 39341, 40806],
        'params': {'dt': 0.1, 'simulation_years': 16, 'init_students': 1000000, 'init_junior': 5000000, 'init_senior': 1000000, 'r': 0.0183, 'K': 13952490, 'alpha': 0.4367, 'school_years': 3, 'base_enrollment': 800000, 'promotion_rate': 0.10, 'natural_turnover': 0.15, 'retirement_rate': 0.08, 'ai_boom_year': 2030, 'ai_speed': 0.4, 'vuln_junior': 0.15, 'vuln_senior': 0.02, 'salary_sensitivity': 0.8, 'enrollment_elasticity': 2.5, 'cost_push_factor': 0.5}
    },
    {
        'name': 'Singer',
        'color': '#d62728', 
        'salary_data': [62857.6, 66019.2, 66435.2, 66768, 67579.2, 69929.6, 71884.8, 74588.8, 78020.8, 83116.8, 86777.6, 86236.8, 104436.8, 107432, 114670.4],
        'params': {'dt': 0.1, 'simulation_years': 16, 'init_students': 10000, 'init_junior': 35000, 'init_senior': 8000, 'r': -0.0189, 'K': 46310, 'alpha': 0.3411, 'school_years': 4, 'base_enrollment': 2000, 'promotion_rate': 0.05, 'natural_turnover': 0.10, 'retirement_rate': 0.04, 'ai_boom_year': 2026, 'ai_speed': 0.9, 'vuln_junior': 0.20, 'vuln_senior': 0.30, 'salary_sensitivity': 1.2, 'enrollment_elasticity': 1.5, 'cost_push_factor': 0.8}
    }
]

# Collection results
results = {}
for config in configs:
    hist_pkg = {'occupation_salary': config['salary_data'], 'social_salary': social_history_data}
    model = OccupationSDModel(config['params'], hist_pkg)
    stocks = model.run()
    
    # Key Metric Calculation
    total_force = stocks['Junior'] + stocks['Senior'] + stocks['Unemployed']
    replacement_rate = stocks['Unemployed'] / (total_force + 1e-9) * 100
    
    active_force = stocks['Junior'] + stocks['Senior']
    junior_ratio = stocks['Junior'] / (active_force + 1e-9) * 100
    
    results[config['name']] = {
        'time': model.time,
        'replacement_rate': replacement_rate,
        'junior_ratio': junior_ratio,
        'color': config['color']
    }

# ==========================================
# PART 3: Draw a Merge Graph (Matplotlib) (AI Statement:This section utilizes Gemini3pro to optimize the generated diagram.)
# ==========================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# --- Subplot 1: Cumulative Replacement Rate ---
for name, data in results.items():
    ax1.plot(data['time'], data['replacement_rate'], color=data['color'], linewidth=3, label=name)

ax1.set_title('Total AI Displacement Severity\n(% of Workforce Replaced)', fontsize=14, fontweight='bold')
ax1.set_ylabel('Cumulative Replacement Rate (%)', fontsize=12)
ax1.set_xlabel('Year', fontsize=12)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
max_replace = max([np.max(d['replacement_rate']) for d in results.values()])
ax1.set_ylim(0, max_replace * 1.1)

# --- Subplot 2: junior_ratio  ---
for name, data in results.items():
    ax2.plot(data['time'], data['junior_ratio'], color=data['color'], linewidth=3, linestyle='-', label=name)

ax2.set_title('Workforce Structure Collapse\n(Share of Junior Roles)', fontsize=14, fontweight='bold')
ax2.set_ylabel('Junior Staff Ratio (%)', fontsize=12)
ax2.set_xlabel('Year', fontsize=12)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, 100)

# Add key annotations
prog_data = results['Programmer']
mid_idx = len(prog_data['time']) // 2 
# ax2.annotate('Junior Collapse!', 
#              xy=(prog_data['time'][mid_idx], prog_data['junior_ratio'][mid_idx]), 
#              xytext=(prog_data['time'][mid_idx]+2, prog_data['junior_ratio'][mid_idx]+15),
#              arrowprops=dict(facecolor='black', shrink=0.05), fontsize=10)

plt.tight_layout()
plt.show()
