import pandas as pd
import numpy as np
# import solver library
import gurobipy as gp
import coptpy as copt

# Slect solver : "copt" or "gurobi"
solver = "copt"

if solver == 'copt':
    BINARY = copt.COPT.BINARY
    CONTINUOUS = copt.COPT.CONTINUOUS
    MINIMIZE = copt.COPT.MINIMIZE
    OPTIMAL = copt.COPT.OPTIMAL
    INFINITY = copt.COPT.INFINITY

elif solver == "gurobi":
    BINARY = gp.GRB.BINARY
    CONTINUOUS = gp.GRB.CONTINUOUS
    MINIMIZE = gp.GRB.MINIMIZE
    OPTIMAL = gp.GRB.OPTIMAL
    INFINITY = gp.GRB.INFINITY

def load_data(dict_path = ""):
    """Reads the data from the given files and returns a dictionary with the data."""
    # 读取Data的数据
    file_path = dict_path + 'Data.xlsx'
    data = pd.ExcelFile(file_path)

    # 读取Result方案的数据，用于计算指标
    # 读取数据
    file_path =  dict_path + 'Result.xlsx'
    result_data = pd.ExcelFile(file_path)
    result_benchmark = result_data.parse('甘特图', index_col=0)

    # Read all necessary sheets
    distance_data = data.parse('待排交路信息', index_col=0)
    train_data = data.parse('车组里程修时信息', index_col=0)
    repair_capability = data.parse('班组检修能力', index_col=0)
    candidate_data = data.parse('候选交路', header=None, index_col=0)
    restore_info = data.parse('车组修后恢复信息', index_col=0)
    day1_schedule = data.parse('Day1检修上线情况', index_col=0)

    # Parameters
    candidate_data.columns = ['1', '2', '3', '4', '5', '6', '7', '8']
    
    reset_distance = {}
    reset_days = {}
    reset_distance['Z'] = restore_info.loc['Z']['修后恢复公里数']
    reset_days['Z'] = restore_info.loc['Z']['修后恢复天数']
    reset_distance['L'] = restore_info.loc['L']['修后恢复公里数']
    
    return candidate_data, distance_data, train_data, repair_capability, reset_distance, reset_days, result_benchmark, day1_schedule

class TrainSchedulePlan:
    def __init__(self, objective_type = "I", candidate_data = None, distance_data = None, train_data = None, repair_capability = None, reset_distance = None, reset_days = None, day1_schedule = None):
        # Parameters
        self.objective_type = objective_type
        self.candidate_data = candidate_data
        self.distance_data = distance_data
        self.train_data = train_data
        self.repair_capability = repair_capability
        self.reset_distance = reset_distance
        self.reset_days = reset_days
        self.day1_schedule = day1_schedule
        self.candidate_routes = {index: set(filter(pd.notna, row[:])) for index, row in candidate_data.iterrows()}
        self.cross_routes = list(distance_data.index)
        self.route_crosses = {key: list(value) for key, value in distance_data.groupby('R_ID').groups.items()}
        self.routes_set =  list(set(distance_data['R_ID'].unique()))
        self.train_set =  list(set(train_data.index.unique()))
        self.days_list = [col.replace('day', '') for col in repair_capability.columns]
        self.num_days = len(self.days_list)
        self.repair_types = list(repair_capability.index)

        # Model
        if solver == 'copt':
            env = copt.Envr()
            self.model = env.createModel()
        elif solver == 'gurobi':
            self.model = gp.Model()
        
        # Decision variables
        # Scheduling variables
        self.x = {}  
        # Maintenance variables
        self.y = {} 
        # Remaining distance variables
        self.z = {}
        # Remaining days variables
        self.d = {}

        self.create_variables()
        self.create_constraints()

    def solve(self):
            if self.objective_type == 'I':
                return self.solve_problem_I()
            elif self.objective_type == 'II':
                return self.solve_problem_II()
            elif self.objective_type == 'III':
                return self.solve_problem_III()
    def evaluate_schedule(self, result_df):
        """
        计算评价指标
        Args:
            result_df (pd.DataFrame): the scheduled trains for each day for each cross routes and repair type.
        
        Returns:
            dict: 评价指标
        """
        def calculate_excess_maintenance(result_df):
            """
            1) 计算过度维修里程和天数
            """
            # 每个车组的初始剩余里程和剩余天数
            remain_distance_days = self.train_data.copy()
            
            # 初始化过度维修余量剩余里程和天数
            remaining_mileage = {type: 0 for type in self.repair_types}
            remaining_days = {type: 0 for type in self.repair_types}

            # 计算剩余里程和天数
            for day in range(1, self.num_days):
                # 车组任务
                for cross in self.cross_routes:
                    train = result_df.loc[cross]['Day' + str(day + 1)]
                    if train == '无任务':
                        continue
                    remain_distance_days.loc[train, 'Z' + '剩余天数'] -= 1
                    for type in self.repair_types:
                        remain_distance_days.loc[train, type + '剩余里程'] -= self.distance_data.loc[cross]['distance']
                # 检修任务
                for type in self.repair_types:
                    if pd.isna(result_df.loc[type]['Day' + str(day + 1)]) or result_df.loc[type]['Day' + str(day + 1)] in ['', None]:
                        continue
                    trains = result_df.loc[type]['Day' + str(day + 1)].split(',')
                    for train in trains:
                        if type == 'Z':
                            remaining_days['Z'] += remain_distance_days.loc[train]['Z剩余天数']
                            remain_distance_days.loc[train, 'Z剩余天数'] = self.reset_days['Z']
                        remaining_mileage[type] += remain_distance_days.loc[train][type + '剩余里程']
                        remain_distance_days.loc[train, type + '剩余里程'] = self.reset_distance[type]
            
            return remaining_mileage, remaining_days
        def calculate_switch_count(df):
            """
            2) 计算换车次数
            """
            train_assigned_routes = {}
            for day in range(self.num_days):
                for cross in self.cross_routes:
                    train = df.loc[cross]['Day' + str(day + 1)]
                    if train == '无任务':
                        continue
                    if train not in train_assigned_routes:
                        train_assigned_routes[train] = []
                        train_assigned_routes[train].append(self.distance_data.loc[cross]['R_ID'])
                    elif self.distance_data.loc[cross]['R_ID'] not in train_assigned_routes[train]:
                        train_assigned_routes[train].append(self.distance_data.loc[cross]['R_ID'])

            switch_count = 0
            for train in train_assigned_routes.keys():
                switch_count += len(train_assigned_routes[train]) - 1
            return switch_count

        def calculate_workload_variance(result_df):
            """
            3) 计算每天维修工作量的极差与方差
            """
            daily_workload = []
            for day in range(self.num_days):
                workload = 0
                for type in self.repair_types:
                    if pd.isna(result_df.loc[type]['Day' + str(day + 1)]) or result_df.loc[type]['Day' + str(day + 1)] in ['', None]:
                        continue
                    trains = result_df.loc[type]['Day' + str(day + 1)].split(',')
                    workload += len(trains)
                daily_workload.append(workload)
            workload_range = max(daily_workload) - min(daily_workload)
            workload_variance = sum([(x - np.mean(daily_workload)) ** 2 for x in daily_workload]) / len(daily_workload)
            return workload_range, workload_variance

        # 计算评价指标
        remaining_mileage, remaining_days = calculate_excess_maintenance(result_df)
        switch_count = calculate_switch_count(result_df)
        workload_range, workload_variance = calculate_workload_variance(result_df)
            
        return {
                '过度维修Z里程': remaining_mileage['Z'],
                '过度维修L里程': remaining_mileage['L'],
                '过度维修Z天数': remaining_days['Z'],
                '累积换车次数': switch_count,
                '维修工作量极差': workload_range,
                '维修工作量方差': workload_variance
            }


    def create_variables(self):
        # create variables
        # x: train assignment to cross route on day : x[train, cross, day] = 1 if train assigned to cross route on day, 0 otherwise
        self.x = {}
        for day in range(self.num_days):
            for train in self.train_set:
                for route in self.candidate_routes[train]:
                    if route not in self.route_crosses.keys():
                        continue
                    for cross in self.route_crosses[route]:
                        if day == 0:
                            value = self.day1_schedule.loc[cross][train]
                            self.x[train, cross, day] = self.model.addVar(vtype = BINARY, lb = value, ub = value, name=f'x_{train}_{cross}_{day + 1}')
                        else:
                            self.x[train, cross, day] = self.model.addVar(vtype= BINARY, name=f'x_{train}_{cross}_{day + 1}')
        # y: train assignment to repair type on day : y[train, type, day] = 1 if train assigned to type on day, 0 otherwise
        # denote: Y_{i, k, t}
        self.y = {}
        for day in range(self.num_days):
            for train in self.train_set:
                for type in self.repair_types:
                    if day == 0:
                        value = self.day1_schedule.loc[type][train]
                        self.y[train, type, day] = self.model.addVar(vtype=BINARY, lb = value, ub = value, name=f'y_{train}_{type}_{day + 1}')
                    else:
                        self.y[train, type, day] = self.model.addVar(vtype=BINARY, name=f'y_{train}_{type}_{day + 1}')
        # z: remaining distance to repair type of train after day : z[train, type, day]
        # d: remaining days to repair type Z of train after day : d[train, day]
        self.z = {}
        for train in self.train_set:
            for type in self.repair_types:
                for day in range(self.num_days):
                    if day == 0:
                        # set solution value of day0
                        remain_distance = self.train_data.loc[train][type + '剩余里程']
                        self.z[train, type, day] = self.model.addVar(vtype=CONTINUOUS, lb=remain_distance, ub=remain_distance, name=f'z_{train}_{type}_{day + 1}')
                    else:  
                        self.z[train, type, day] = self.model.addVar(vtype=CONTINUOUS, lb=0, ub=self.reset_distance[type], name=f'z_{train}_{type}_{day + 1}')
        self.d = {}
        for train in self.train_set:
            for day in range(self.num_days):
                if day == 0:
                    # set solution value of day0
                    remain_Z_days = self.train_data.loc[train]['Z剩余天数']
                    self.d[train, day] = self.model.addVar(vtype=CONTINUOUS, lb=remain_Z_days, ub=remain_Z_days, name=f'd_{train}_{day}')
                else:
                    self.d[train, day] = self.model.addVar(vtype=CONTINUOUS, lb=0, ub=self.reset_days['Z'], name=f'd_{train}_{day}')


    def create_constraints(self):
        # Add constraints
        # 1）每日各检修项目检修能力限制，安排检修计划不能突破检修能力：\sum\limits_{i \in I} Y_{i, k, t} \leq C_{k, t} \for k, t.
        for day in range(self.num_days):
            self.model.addConstr(sum(self.y[train, 'Z', day] for train in self.train_set) <= self.repair_capability.loc['Z']['day' + str(day + 1)],
                            name=f'ZRepairCapacity_{day + 1}')
            self.model.addConstr(sum(self.y[train, 'L', day] for train in self.train_set) <= self.repair_capability.loc['L']['day' + str(day + 1)],
                            name=f'LRepairCapacity_{day + 1}')

        # 2) 运检计划不能有交路缺编，即没有车组执行：sum_{i} X[i, c, t] == 1  \for i,t
        for route in self.routes_set:
            for cross in self.route_crosses[route]:
                for day in range(self.num_days):
                    if self.distance_data.loc[cross]['day' + str(day + 1)] == 1:
                        self.model.addConstr(sum(self.x[train, cross, day] 
                                            for train in self.train_set if route in self.candidate_routes[train]) 
                                            == 1,
                                    name=f'CrossAssignment_{cross}_{day + 1}')

        # 3) 车组状态：同一时刻同一车组至多安排一个交路任务，或被检修
        for train in self.train_set:
            for day in range(self.num_days):
                for type in ['Z', 'L']:
                    self.model.addConstr(
                        sum(self.x[train, cross, day]
                                    for route in self.candidate_routes[train]
                                    if route in self.route_crosses.keys()
                                    for cross in self.route_crosses[route]
                                    ) <= 1 - self.y[train, type, day],
                        name=f'TrainState_{train}_{type}_{day + 1}')
                    # 是否可以同时执行多种检修？？
                    # 否
                    # self.model.addConstr(
                    #     sum(y[train, type, day] for type in ['Z', 'L']) <= 1,
                    #     name=f'TrainRepairState_{train}_{day}')

        # 4) 每辆车的每种检修项目的剩余可用里程数和天数不能突破
        for train in self.train_set:
            for day in range(1, self.num_days):
                # update the remaining mileage and day
                # distances
                # Z
                self.model.addConstr(self.z[train, 'Z', day] >= self.z[train, 'Z', day - 1] - sum(self.x[train, cross, day] * self.distance_data.loc[cross]['distance']
                                                                                for route in self.candidate_routes[train]
                                                                                if route in self.route_crosses.keys()
                                                                                for cross in self.route_crosses[route]
                                                                                ),
                                name=f'ZRemainingDistance_{train}_{day + 1}_lb')
                self.model.addConstr(self.z[train, 'Z', day] <= self.z[train, 'Z', day - 1] - sum(self.x[train, cross, day] * self.distance_data.loc[cross]['distance']
                                                                                for route in self.candidate_routes[train]
                                                                                if route in self.route_crosses.keys()
                                                                                for cross in self.route_crosses[route]
                                                                                ) + self.reset_distance['Z'] * self.y[train, 'Z', day],
                                name=f'ZRemainingDistance_{train}_{day + 1}_ub')
                # L
                self.model.addConstr(self.z[train, 'L', day] >= self.z[train, 'L', day - 1] - sum(self.x[train, cross, day] * self.distance_data.loc[cross]['distance']
                                                                                for route in self.candidate_routes[train]
                                                                                if route in self.route_crosses.keys()
                                                                                for cross in self.route_crosses[route]
                                                                                ),
                                name=f'LRemainingDistance_{train}_{day + 1}_lb')
                self.model.addConstr(self.z[train, 'L', day] <= self.z[train, 'L', day - 1] - sum(self.x[train, cross, day] * self.distance_data.loc[cross]['distance']
                                                                                for route in self.candidate_routes[train]
                                                                                if route in self.route_crosses.keys()
                                                                                for cross in self.route_crosses[route]
                                                                                ) + self.reset_distance['L'] * self.y[train, 'L', day],
                                name=f'LRemainingDistance_{train}_{day + 1}_ub')
                # days
                # Z
                self.model.addConstr(self.d[train, day] >= self.d[train, day - 1] - sum(self.x[train, cross, day] 
                                                                        for route in self.candidate_routes[train]
                                                                        if route in self.route_crosses.keys()
                                                                        for cross in self.route_crosses[route]
                                                                        ),
                                name=f'RemainingDays_{train}_{day + 1}_lb')
                self.model.addConstr(self.d[train, day] <= self.d[train, day - 1] - sum(self.x[train, cross, day] 
                                                                        for route in self.candidate_routes[train]
                                                                        if route in self.route_crosses.keys()
                                                                        for cross in self.route_crosses[route]
                                                                        ) + self.reset_days['Z'] * self.y[train, 'Z', day],
                                name=f'RemainingDays_{train}_{day + 1}_ub')

        # 5) R_ID相同的交路需要从上到下连续执行
        for train in self.train_set:
            for route in self.candidate_routes[train]:
                if route  not in self.route_crosses.keys():
                    continue
                for cross_index in range(1, len(self.route_crosses[route])):
                        cross = self.route_crosses[route][cross_index]
                        pre_cross = self.route_crosses[route][cross_index - 1]
                        for day in range(1, self.num_days): 
                            # x[i,r,t] == x[i,r-1,t-1]
                            self.model.addConstr(self.x[train, cross, day] == self.x[train, pre_cross, day - 1],
                                            name=f'Continuity_{train}_{cross}_{day + 1}')
                        
    def get_solution(self):
        """
        提取schedule
        """
        columns = ['Day1', 'Day2', 'Day3', 'Day4', 'Day5', 'Day6', 'Day7', 'Day8']
        result_df = pd.DataFrame(index=self.cross_routes + self.repair_types + ['后备车组'], columns=columns)

        # 将 NaN 值替换为 '无任务'
        result_df.fillna('无任务', inplace=True)

        # cross route schedule
        flags = {}
        for day in range(self.num_days):
            for train in self.train_set:
                flags[train, day] = False
                for route in self.candidate_routes[train]:
                    if route not in self.route_crosses.keys():
                            continue
                    for cross in self.route_crosses[route]:
                        if self.x[train, cross, day].X > 0.5:
                                flags[train, day] = True
                                result_df.loc[cross, f'Day{day+1}'] = f'{train}'

        # repair type schedule
        for day in range(self.num_days):
            for type in self.repair_types:
                for train in self.train_set:
                    if result_df.loc[type, f'Day{day+1}'] == '无任务':
                        result_df.loc[type, f'Day{day+1}'] = ""
                    if self.y[train, type, day].X > 0.5:
                        flags[train, day] = True
                        if result_df.loc[type, f'Day{day+1}'] != "":
                            result_df.loc[type, f'Day{day+1}'] += ","
                        result_df.loc[type, f'Day{day+1}'] += (f'{train}')

        # remaining trains
        for day in range(self.num_days):
            if result_df.loc['后备车组', f'Day{day+1}'] == '无任务':
                result_df.loc['后备车组', f'Day{day+1}'] = ""
            for train in self.train_set:
                if flags[train, day] == False:
                    if result_df.loc['后备车组', f'Day{day+1}'] != "":
                        result_df.loc['后备车组', f'Day{day+1}'] += ","     
                    result_df.loc['后备车组', f'Day{day+1}'] += (f'{train}')

        # print(result_df)

        return result_df


    def solve_problem_I(self):
        """
        求解问题I：避免过修，即检修间隔尽量用足剩余里程和天数
        """
        # 1) 引入辅助变量
        # z'[i, k, t]
        # d'[i, t]
        zz = {}
        dd = {}
        for day in range(self.num_days):
            for train in self.train_set:
                dd[train, day] = self.model.addVar(vtype=CONTINUOUS, lb=0, ub= self.reset_days['Z'], name=f'dd_{train}_Z_{day + 1}')
                for type in self.repair_types:
                    zz[train, type, day] = self.model.addVar(vtype=CONTINUOUS, lb=0, ub= self.reset_distance[type], name=f'zz_{train}_{type}_{day + 1}')
        
        # 2) 添加辅助约束
        #  z'[i, k, t] = z[i, k, t] * y[i, k, t]  <==>  z'[i, k, t] >= z[i, k, t] - reset_distance[t] * (1 - y[i, k, t])
        #  d'[i, t] = sum{k} d[i, t] * y[i, 'Z', t] <==>  d'[i, t] >= sum{k} d[i, t] - reset_distance[t] * (1 - y[i, 'Z', t])
        for day in range(self.num_days):
            for train in self.train_set:
                self.model.addConstr(dd[train, day] >= self.d[train, day] - self.reset_days['Z'] * (1 - self.y[train, 'Z', day]), name=f'dd_{train}_Z_{day + 1}')
                for type in self.repair_types:
                    self.model.addConstr(zz[train, type, day] >= self.z[train, type, day] - self.reset_distance[type] * (1 - self.y[train, type, day]), name=f'zz_{train}_{type}_{day + 1}')
                    
        # 3) 设置目标函数
        # min \sum{i, k, t} z'[i, k, t] + \sum{i, t} d'[i, t]
        objective1 = sum(zz[train, type, day] 
                        for train in self.train_set 
                        for type in self.repair_types 
                        for day in range(self.num_days)) + sum(dd[train, day] 
                                                        for train in self.train_set 
                                                        for day in range(self.num_days))
        self.model.setObjective(objective1, sense=MINIMIZE)

        # 4) 求解模型
        # Optimize self.model
        # self.model.write("Objective1.lp")
        if solver == 'copt':
            self.model.solve()
        elif solver == 'gurobi':
            self.model.optimize()
        print("Objective1: %g" % self.model.objVal)

        # 5) 提取Schedule，输出结果
        result1 = self.get_solution()
        metrics_result1 = self.evaluate_schedule(result1)
        # print(metrics_result1)

        return result1, metrics_result1

    def solve_problem_II(self):
        """
        求解问题II：换车次数少，希望车组V执行连续交路R之后，继续再次执行连续交路R
        """
        # 1) 引入辅助变量：xx[train, route] = max{x[train, cross, day] for cross in cross_routes[route]}, 1 表示车组V在第day天执行了连续交路R
        xx = {}
        for train in self.train_set:
            for route in self.candidate_routes[train]:
                for day in range(self.num_days):
                    xx[train, route] = self.model.addVar(vtype=BINARY, name=f'xx_{train}_{route}')
        
        # 2) 添加辅助约束
        # xx[train, route] = max{x[train, cross, day]  <==> xx[train, route, day] >= x[train, cross, day] for cross in cross_routes[route]
        for train in self.train_set:
            for route in self.candidate_routes[train]:
                if route in self.route_crosses.keys():
                    for cross in self.route_crosses[route]:
                        for day in range(self.num_days):
                            self.model.addConstr(xx[train, route] >= self.x[train, cross, day], name=f'xx_{train}_{route}_{day + 1}')
    
        # 3) 设置目标函数
        # min sum_{i, r} xx[i, r]
        objective2 = sum(xx[train, route] 
                        for train in self.train_set 
                        for route in self.candidate_routes[train])
                            
        self.model.setObjective(objective2, sense=MINIMIZE)
        

        # 4) 求解模型
        # Optimize self.model
        # self.model.write("Objective2.lp")

        if solver == 'copt':
            self.model.solve()
        elif solver == 'gurobi':
            self.model.optimize()
        print("Objective2: %g" % self.model.objVal)


        # 5) 提取Schedule，输出结果
        result2 = self.get_solution()
        metrics_result2 = self.evaluate_schedule(result2)
        # print(metrics_result2)

        return result2, metrics_result2

    def solve_problem_III(self):
        """
        求解问题III：每日检修工作量均匀
        最小化检修工作量极差 min(max_t(sum_{i, k}y[i,k,t]) - min_t(sum_{i, k}y[i,k,t]))
        """
        # 1) 引入辅助变量: 
        #  yy_max = max_t(sum_{i, k}y[i,k,t])
        #  yy_min = min_t(sum_{i, k}y[i,k,t])
        yy_max = self.model.addVar(vtype=CONTINUOUS, lb = 0, ub = INFINITY, name = "yy_max")
        yy_min = self.model.addVar(vtype=CONTINUOUS, lb = 0, ub = INFINITY, name = "yy_min")
        
        # 2) 添加辅助约束
        #   yy_max = max_t(sum_{i, k}y[i,k,t])   <==>  yy_max >= sum_{i, k}y[i,k,t] for all t
        #   yy_min = min_t(sum_{i, k}y[i,k,t])   <==>  yy_min <= sum_{i, k}y[i,k,t] for all t
        for day in range(self.num_days):
            self.model.addConstr(yy_max >= sum(self.y[train, type, day] for train in self.train_set for type in self.repair_types), name = "yy_max_%d"%day)
            self.model.addConstr(yy_min <= sum(self.y[train, type, day] for train in self.train_set for type in self.repair_types), name = "yy_min_%d"%day)
    
        # 3）设置目标函数
        #  min yy_max - yy_min
        objective3 = yy_max - yy_min
        self.model.setObjective(objective3, sense=MINIMIZE)
        
        # 4) 求解模型
        # Optimize model
        # self.model.write("Objective3.lp")

        if solver == 'copt':
            self.model.solve()
        elif solver == 'gurobi':
            self.model.optimize()

        print("Objective3: %g" % self.model.objVal)

        # 5) 提取Schedule，输出结果
        result3 = self.get_solution()
        metrics_result3 = self.evaluate_schedule(result3)
        # print(metrics_result3)
        
        return result3, metrics_result3

if __name__ == "__main__":
    candidate_data, distance_data, train_data, repair_capability, reset_distance, reset_days, result_benchmark, day1_schedule = load_data("")

    tspI = TrainSchedulePlan(objective_type="I", 
                             candidate_data = candidate_data, 
                             distance_data = distance_data, 
                             train_data = train_data, 
                             repair_capability=repair_capability, 
                             reset_distance=reset_distance, 
                             reset_days=reset_days,
                             day1_schedule = day1_schedule
                             )
    tspII = TrainSchedulePlan(objective_type="II", 
                             candidate_data = candidate_data, 
                             distance_data = distance_data, 
                             train_data = train_data, 
                             repair_capability=repair_capability, 
                             reset_distance=reset_distance, 
                             reset_days=reset_days,
                             day1_schedule = day1_schedule
                             )
    tspIII = TrainSchedulePlan(objective_type="III", 
                             candidate_data = candidate_data, 
                             distance_data = distance_data, 
                             train_data = train_data, 
                             repair_capability=repair_capability, 
                             reset_distance=reset_distance, 
                             reset_days=reset_days,
                             day1_schedule = day1_schedule
                             )
    # 对比评价指标
    metrics_benchmark = tspI.evaluate_schedule(result_benchmark)
    print("Benchmark: ", metrics_benchmark)
    # Objective 1
    result1, metrics_result1 =  tspI.solve()
    print("Objective I: ", metrics_result1)
    # Objective 2
    result2, metrics_result2 =  tspII.solve()
    print("Objective II: ", metrics_result2)
    # Objective 3
    result3, metrics_result3 =  tspIII.solve()
    print("Objective III: ", metrics_result3)

    metrics_df = pd.DataFrame([metrics_benchmark, metrics_result1, metrics_result2, metrics_result3], index=['Benchmark', 'Objective I', 'Objective II', 'Objective III'])

    # 将结果写入到Excel文件的不同Sheet中
    with pd.ExcelWriter('optimization_results.xlsx') as writer:
        result_benchmark.to_excel(writer, sheet_name='甘特图(benchmark)')
        result1.to_excel(writer, sheet_name='甘特图-I')
        result2.to_excel(writer, sheet_name='甘特图-II')
        result3.to_excel(writer, sheet_name='甘特图-III')
        metrics_df.to_excel(writer, sheet_name='评价指标')