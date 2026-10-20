import random
import time
import json
import gymnasium as gym
import torch
import os, sys
from pprint import pprint
from collections import defaultdict


# Generate training data
class CaseGenerator:
    """
    FJSP instance generator
    """

    def __init__(
        self,
        num_jobs,
        num_mas,
        opes_per_job_min=10,
        opes_per_job_max=20,
        nums_oper_per_job: list | None = None,
        path="../data/data_train/",
        flag_same_opes=True,
        save_doc=False,
    ):

        # Whether to save the generated instances to files
        self.save_doc = save_doc

        # Whether each generated instance uses a fixed or random number of operations per job
        self.flag_same_opes = flag_same_opes

        # Number of operations per job
        self.num_oper_per_job = [] if nums_oper_per_job is None else nums_oper_per_job
        self.path = path

        # Number of jobs and machines
        self.num_jobs = num_jobs
        self.num_machines = num_mas

        # Minimum and maximum number of eligible machines for an operation
        self.mas_per_ope_min = 1
        self.mas_per_ope_max = min(num_mas, 3)  # Limit the number of machines that can process one operation

        # Minimum and maximum number of operations in a job
        self.opes_per_job_min = opes_per_job_min
        self.opes_per_job_max = opes_per_job_max

        # Minimum and maximum processing times for each operation
        self.high_freq_proctime_min = 1
        self.high_freq_proctime_max = 60
        self.low_freq_proctime = [_ for _ in range(1, 11)] * 60

        # Factor used with the bounds above to generate operation processing times
        self.proctime_dev = 0.2

        # Fixed list of power values to sample from
        self.power_candidates = [100 * i for i in range(0, 20)] + [500 * i for i in range(5, 15)]

    def get_case(self, idx=0, weight_prob=1):
        """
        Generate FJSP instance
        :param idx: The instance number
        """

        # If self.flag_same_opes is false, randomly generate the nums_oper array
        if not self.flag_same_opes:
            self.num_oper_per_job = [
                random.randint(self.opes_per_job_min, self.opes_per_job_max) for _ in range(self.num_jobs)
            ]
        # Total number of operations
        self.num_of_all_opers = sum(self.num_oper_per_job)

        # Number of eligible machines for each operation
        self.num_option_per_oper = [
            random.randint(self.mas_per_ope_min, self.mas_per_ope_max) for _ in range(self.num_of_all_opers)
        ]

        # Total number of eligible machine assignments; dividing this by self.num_opes gives the average
        # number of eligible machines per operation, usually listed in the third column of the first data row
        self.num_options = sum(self.num_option_per_oper)

        # This stores the eligible machine assignments for each operation
        self.oper_machine_pair = []
        # In the loop, option is the number of eligible machines for each operation
        for option in self.num_option_per_oper:
            # random.sample(range(1, self.num_machines+1), option) randomly selects option eligible machines from all machines for this operation
            # Sort them for easier selection; the '+' operator concatenates two lists
            self.oper_machine_pair = self.oper_machine_pair + sorted(
                random.sample(range(1, self.num_machines + 1), option)
            )

        # Compute the mean processing time for each operation, then use it to generate times on each machine
        prob_high_freq = 0.8
        self.proc_times_mean = [
            (
                random.randint(self.high_freq_proctime_min, self.high_freq_proctime_max)
                if random.random() < prob_high_freq
                else random.choice(self.low_freq_proctime)
            )
            for _ in range(self.num_of_all_opers)
        ]

        # Processing time for each operation
        self.proc_time = []
        for i in range(self.num_of_all_opers):
            low_bound = max(
                self.high_freq_proctime_min,
                round(self.proc_times_mean[i] * (1 - self.proctime_dev)),
            )
            high_bound = round(self.proc_times_mean[i] * (1 + self.proctime_dev))
            # Processing time for this operation on each machine; concatenate as for self.ope_ma
            proc_time_ope = [
                random.randint(low_bound, high_bound) for _ in range(self.num_option_per_oper[i])
            ]
            self.proc_time.extend(proc_time_ope)

        # Index of the first operation in each job
        self.num_ope_biases = [sum(self.num_oper_per_job[0:i]) for i in range(self.num_jobs)]
        # Cumulative number of eligible machines across operations
        self.num_ma_biass = [sum(self.num_option_per_oper[0:i]) for i in range(self.num_of_all_opers)]

        # Standard dataset format
        # Row 0: number of jobs, number of machines, average eligible machines per operation
        line0 = f"{self.num_jobs}\t{self.num_machines}\t{self.num_options / self.num_of_all_opers}"
        # Row 1: capacity of each machine
        line1 = "\t".join(["{}".format(random.randint(1, 10)) for _ in range(self.num_machines)])
        # Row 2: power of each machine (unit: W)
        line2 = "\t".join(
            ["{}".format(random.choice(self.power_candidates)) for _ in range(self.num_machines)]
        )

        lines = []
        lines.append(line0 + "\n")
        lines.append(line1 + "\n")
        lines.append(line2 + "\n")

        # Generate each job
        for i in range(self.num_jobs):
            flag = 0
            flag_time = 0
            flag_new_oper = 1
            idx_ope_this_job = -1
            idx_ma = 0
            line = []





            option_max = sum(
                self.num_option_per_oper[
                    self.num_ope_biases[i] : (self.num_ope_biases[i] + self.num_oper_per_job[i])
                ]
            )
            idx_option = 0

            while True:
                if flag == 0:
                    # flag = 0: start generating a job row; the first number is the job's operation count
                    line.append(self.num_oper_per_job[i])
                    flag += 1
                elif flag == flag_new_oper:
                    # Start generating a new operation in this job row
                    idx_ope_this_job += 1
                    idx_ma = 0
                    flag_new_oper += (
                        self.num_option_per_oper[self.num_ope_biases[i] + idx_ope_this_job] * 2 + 1
                    )
                    # Generate the number of eligible machines for this new operation
                    line.append(self.num_option_per_oper[self.num_ope_biases[i] + idx_ope_this_job])
                    flag += 1
                elif flag_time == 0:
                    # Generate an eligible machine index
                    line.append(
                        self.oper_machine_pair[
                            self.num_ma_biass[self.num_ope_biases[i] + idx_ope_this_job] + idx_ma
                        ]
                    )
                    flag += 1
                    flag_time = 1
                else:
                    # Generate the processing time of this operation on that machine
                    line.append(
                        self.proc_time[self.num_ma_biass[self.num_ope_biases[i] + idx_ope_this_job] + idx_ma]
                    )
                    flag += 1
                    flag_time = 0
                    idx_option += 1
                    idx_ma += 1

                # If all eligible machines for this job have been generated, exit and start the next job
                if idx_option == option_max:
                    lines.append(" ".join(map(str, line)) + "\n")
                    break

        # Generate operation parameters: -1 means none; the first and last operation of each job have none
        for i in range(self.num_jobs):

            choice_len = self.num_of_all_opers + 1 - self.num_jobs * 2
            options = [-1] + list(range(1, choice_len + 1))
            weights = [choice_len * weight_prob] + [1] * choice_len  # Probability weights
            line = [-1] + random.choices(options, weights, k=self.num_oper_per_job[i] - 2) + [-1]
            lines.append(" ".join(map(str, line)) + "\n")

        # Generate seamless links to successor operations: 1 means seamless and 0 means not; the first and last operations are 0
        # Each job has at most one seamless segment, with length between 2 and half the operation count
        for i in range(self.num_jobs):
            n = self.num_oper_per_job[i]
            if n < 4:
                line = [0] * n
            else:
                seg_len = random.randint(2, n // 2)
                start = random.randint(1, n - seg_len - 1)  # This ensures the first and last entries are 0
                line = [0] * start + [1] * seg_len + [0] * (n - start - seg_len)
            lines.append(" ".join(map(str, line)) + "\n")

        lines.append("\n")

        if self.save_doc:
            file_name = f"{self.num_jobs:02}{self.num_machines:02}/{self.num_jobs}j_{self.num_machines}m_{idx + 1:03}.fjs"
            file_path = self.path + file_name
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "w") as f:
                f.writelines(lines)

        lines_dict = self.convert_fjs_to_json(lines)

        return lines, lines_dict

    # Get the number of jobs, machines, and total operations in this instance
    @staticmethod
    def fetch_basic_info(task_info_dict):
        return (
            task_info_dict["info"]["num_jobs"],
            task_info_dict["info"]["num_machines"],
            task_info_dict["info"]["num_opers"],
        )

    # Read the JSON data directly
    @staticmethod
    def convert_json_to_json(task_info):
        for k, v in task_info["detail"].items():
            for i in range(len(v)):
                v[i] = (
                    v[i][0],
                    [v[i][1]] * len(v[i][0]),
                    v[i][2],
                    v[i][3],
                )

        # Create mapping tables
        job_mapping = {job: idx for idx, job in enumerate(task_info["detail"].keys())}
        machine_names = set(task_info["info"]["machine_capacities"].keys())

        # RL requires machine indices to start at 1
        machine_mapping = {machine: idx + 1 for idx, machine in enumerate(machine_names)}

        # Parameter mapping table
        param_mapping = {}
        param_counter = 1  # Regular parameters start at 1; -1 means no parameter

        mappings = {
            "job_mapping": job_mapping,
            "reverse_job_mapping": {idx: job for job, idx in job_mapping.items()},
            "machine_mapping": machine_mapping,
        # Note: FJSP-B and RL both subtract 1 from machine IDs; subtract 1 when using mappings to restore machine names
            "reverse_machine_mapping": {idx - 1: machine for machine, idx in machine_mapping.items()},
            "param_mapping": {},  # Store parameter mappings
            "reverse_param_mapping": {},  # Reverse parameter mappings
        }

        mapped_task_info = {
            "detail": {},
            "info": {
                "machine_capacities": {},
                "num_jobs": None,
                "num_machines": None,
                "num_opers": None,
                "machines_available_time": None,
                "job_lock": None,
            },
        }

        mapped_task_info["info"]["num_jobs"] = len(job_mapping)
        mapped_task_info["info"]["num_machines"] = len(machine_mapping)
        mapped_task_info["info"]["num_opers"] = sum([len(v) for _, v in task_info["detail"].items()])

        machine_capacities = {
            machine_mapping[m]: cap for m, cap in task_info["info"]["machine_capacities"].items()
        }
        mapped_task_info["info"]["machine_capacities"] = [
            machine_capacities[key] for key in sorted(machine_capacities)
        ]
        mapped_task_info["info"]["machines_available_time"] = [0 for _ in range(len(machine_capacities))]

        for job, tasks in task_info["detail"].items():
            mapped_job = job_mapping[job]
            mapped_task_info["detail"][mapped_job] = []
            for machine_list, processtimes, machine_details, extra_info in tasks:
                mapped_machine_list = [machine_mapping[m] for m in machine_list]

                # Map parameters using only the param from the first machine_details entry
                first_param = machine_details[0]["param"].copy() if machine_details else {}
                first_param.pop("bottle_count", None)
                param_str = str(first_param) if first_param else "{}"  # Handle an empty dictionary

                if param_str not in param_mapping:
                    param_mapping[param_str] = -1 if first_param == {} else param_counter
                    mapped_param = param_mapping[param_str]
                    if first_param != {}:
                        param_counter += 1
                else:
                    mapped_param = param_mapping[param_str]  # Reuse the existing index

                mapped_task_info["detail"][mapped_job].append(
                    (mapped_machine_list, processtimes, mapped_param)
                )

        mappings["param_mapping"] = param_mapping
        mappings["reverse_param_mapping"] = {v: k for k, v in param_mapping.items()}

        return task_info, mappings, mapped_task_info

    # Convert the standard dataset to JSON and save it
    @staticmethod
    def convert_fjs_to_json(fjs_data):
        task_info = defaultdict(dict)

        # Parse row 1: number of jobs, number of machines, and average eligible machines per operation
        header = fjs_data[0].strip().split()
        num_jobs = int(header[0])

        # Rows 2 and 3 contain machine capacities and power values, respectively
        machine_capacities = list(map(int, fjs_data[1].strip().split()))
        machine_power_info = list(map(int, fjs_data[2].strip().split()))

        # Start from row 3
        num_opers = 0
        for i in range(3, 3 + num_jobs):
            num_opers += int(fjs_data[i].strip().split()[0]) if fjs_data[i] != "\n" else 0

        task_info["info"].update(
            {
                "num_jobs": int(header[0]),
                "num_machines": int(header[1]),
                "num_opers": num_opers,
                "machine_capacities": machine_capacities,
                "mean_option_per_oper": float(header[2]),
                "machines_available_time": [0 for _ in range(len(machine_capacities))],
                "machine_power_info": machine_power_info,
            }
        )

        # Parse the data for each job
        job_lock = {}
        for job_idx in range(num_jobs):
            # Get the data row for the current job
            job_data = fjs_data[3 + job_idx].strip().split()
            opers_param = fjs_data[3 + num_jobs + job_idx].strip().split()
            job_lock[job_idx] = [int(x) for x in fjs_data[3 + 2 * num_jobs + job_idx].strip().split()]
            job_operations = []

            # Parse the number of operations
            num_oper_this_job = int(job_data[0])
            ptr = 1  # Pointer used to traverse job_data

            # Parse each operation
            for i in range(num_oper_this_job):
                # Get the number of eligible machines
                num_options = int(job_data[ptr])
                ptr += 1

                # Parse eligible machines and their processing times: ([machine], [processing_time], params)
                options = ([], [], int(opers_param[i]))
                for _ in range(num_options):
                    machine = int(job_data[ptr])
                    processing_time = int(job_data[ptr + 1])
                    options[0].append(machine)
                    options[1].append(processing_time)
                    ptr += 2

                # Add the operation to the current job's operation list
                job_operations.append(options)

            # Add the current job's operation list to the result dictionary
            task_info["detail"][job_idx] = job_operations

        task_info["info"].update({"job_lock": job_lock})
        return task_info


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate FJSP instances.")
    parser.add_argument("--num_examples", type=int, required=True, help="Number of examples")
    parser.add_argument("--num_jobs", type=int, default=10, help="Number of jobs")
    parser.add_argument("--num_machines", type=int, default=10, help="Number of machines")
    parser.add_argument("--opes_per_job_min", type=int, default=10)
    parser.add_argument("--opes_per_job_max", type=int, default=20)
    parser.add_argument("--path", type=str, default="data_train", help="Path to save generated instances")
    parser.add_argument("--save", action="store_true", help="")
    args = parser.parse_args()

    num_jobs = args.num_jobs
    num_machines = args.num_machines
    opes_per_job_min = args.opes_per_job_min
    opes_per_job_max = args.opes_per_job_max
    path = args.path

    gen_groups = [(3, 4), (10, 20), (15, 20), (20, 20)]

    for num_jobs, num_machines in gen_groups:
        start = time.time()
        case = CaseGenerator(
            num_jobs,
            num_machines,
            opes_per_job_min,
            opes_per_job_max,
            nums_oper_per_job=[],
            path=f"./data/{path}/",
            flag_same_opes=False,
            save_doc=args.save,
        )
        for i in range(0, args.num_examples):
            if 5 * i < args.num_examples:
                lines, lines_dict = case.get_case(idx=i, weight_prob=4)
            elif 5 * i < 2 * args.num_examples:
                lines, lines_dict = case.get_case(idx=i, weight_prob=3)
            elif 5 * i < 3 * args.num_examples:
                lines, lines_dict = case.get_case(idx=i, weight_prob=2)
            elif 5 * i < 4 * args.num_examples:
                lines, lines_dict = case.get_case(idx=i, weight_prob=1)
            elif 5 * i < 5 * args.num_examples:
                lines, lines_dict = case.get_case(idx=i, weight_prob=0.5)
            else:
                lines, lines_dict = case.get_case(idx=i, weight_prob=1)






        print(f"Gen J{num_jobs} M{num_machines} cases spend time: {time.time() - start: .3f}s")
