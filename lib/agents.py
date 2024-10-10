import json
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel


class ConfigProxy:
    def __init__(self, config):
        self.__dict__ = config
        self.logs = ""

    def update(self, config):
        self.__dict__.update(config)

    def get_by_name(self, name):
        return self.__dict__.get(name)

    def set_by_name(self, name, value):
        self.__dict__[name] = value

    def clear_logs(self):
        self.logs = ""

    def print_logs(self, log):
        self.logs += str(log) + "\n"

    def print_header(self, header):
        self.logs += f"\n{header:-^80}\n"

    def reset(self):
        self.__dict__ = {}


class PlanFormat(BaseModel):
    thoughts: list[str]
    plan: list[str]


class PlanStepFormat(BaseModel):
    thoughts: list[str]
    options: list[str]


class PlanChoiceFormat(BaseModel):
    thoughts: list[str]
    chosen_option: str
    chosen_option_index: int
    is_terminating: bool


class SelfConsistencyFormat(BaseModel):
    thoughts: list[str]
    chosen_plan: str
    chosen_plan_index: int


class PlanReflector(BaseModel):
    thoughts: list[str]
    changed: bool
    changes_list: list[str]
    plan: str


class FreeLambda:
    def __init__(self, hook, output_name, print_name=None):
        self.hook = hook
        self.output_name = output_name
        self.print_name = str(id(self)) if print_name is None else print_name

    def invoke(self, config):
        result = self.hook(config)
        config.set_by_name(self.output_name, result)


class LLMLambda:
    def __init__(
        self,
        context,
        model,
        structure=None,
        need_logprobs=False,
        post_hook=None,
        output_name="",
        print_name=None,
    ):
        self.context = context
        self.model = model
        self.structure = structure
        self.need_logprobs = need_logprobs
        self.post_hook = post_hook
        self.output_name = output_name
        self.print_name = str(id(self)) if print_name is None else print_name
        self.print_name += " (Model: " + self.model.name + ")"

    def invoke(self, config):
        config.print_header(f"LLM Lambda {self.print_name}")
        context = self.context.format_map(config.__dict__)
        config.print_logs(f"Query: {context}")
        result, _, probs, _  = self.model.model_response(
                context, structure=self.structure, need_logprobs=self.need_logprobs
            )
        result = self.model.get_text_from_message(result)
        if self.need_logprobs:
            config.set_by_name(self.output_name + "_probs", probs)
        config.set_by_name(self.output_name, result)
        if self.post_hook is not None:
            self.post_hook(config)
        config.print_logs("Response: \n" + str(result))


class Planner:
    def __init__(
        self,
        context,
        action_space,
        output_name,
        model,
        self_consistency_rounds=None,
        algorithm="",
        print_name=None,
        tot_rounds=0,
        tot_number=0,
    ):
        self.context = context
        self.action_space = action_space
        self.self_consistency_rounds = (
            1 if self_consistency_rounds is None else self_consistency_rounds
        )
        self.algorithm = algorithm
        self.output_name = output_name
        self.print_name = str(id(self)) if print_name is None else print_name
        self.model = model
        self.tot_rounds = tot_rounds
        self.tot_number = tot_number

    def invoke(self, config):
        config.print_header(f"Planner {self.print_name}")
        config.set_by_name("probs", [])
        match self.algorithm:
            case "cot":
                plan_constructor = self.cot_planner
            case "tot":
                plan_constructor = self.tot_planner
        if self.self_consistency_rounds == 1:
            final_plan = plan_constructor(config)
        else:
            plans = []
            for round in range(self.self_consistency_rounds):
                plans.append(plan_constructor(config))
            final_plan = self.self_consistency_choice(config, plans)
        config.set_by_name(self.output_name, final_plan)

    def self_consistency_choice(self, config, plans):
        plans_str = "\n\n".join([f"Plan {n}:\n{plan}" for n, plan in enumerate(plans)])
        context = self.context.format_map(config.__dict__)
        planner_cot_voter_prompt = f"""
        {context}
        
        You will be given a list of plans.
        You should find the best plan and print it.

        Your answer should be in the following form:
        {{
            thoughts: *your thoughts, think step by step*
            chosen_plan: *best plan*
            chosen_plan_index: *index of best plan*
        }}
        
        Plans:
        {plans_str}
        """
        config.print_logs(f"Prompt: {planner_cot_voter_prompt}")
        result, _ = self.model.single_query_response(
            planner_cot_voter_prompt, structure=SelfConsistencyFormat
        )
        result = self.model.get_structure_from_message(result)
        config.print_logs(f"Response: {result}")
        return plans[result.chosen_plan_index]

    def cot_planner(self, config):
        context = self.context.format_map(config.__dict__)
        planner_prompt = f"""
        {context}

        You can do these and only these actions as steps:
        {"- \n".join(self.action_space)}

        Do not give any examples, only plan for the task, be precise. Think step by step.
        You can use only steps written above!

        Your answer should be in the following form:
        {{
            thoughts: *your thoughts, think step by step*
            plan: *plan by possible steps*
        }}
        """
        config.print_logs(f"Prompt: {planner_prompt}")
        result, _, probs = self.model.single_query_response(
            planner_prompt, structure=PlanFormat, need_logprobs=True
        )
        result = self.model.get_structure_from_message(result)
        config.probs.append(probs)
        config.print_logs(f"Response: \n{result}")
        return (
            "Empty"
            if len(result.plan) == 0
            else "\n".join([f"{n}: {el}" for n, el in enumerate(result.plan)])
        )

    def tot_planner(self, config):
        current_plan = []
        current_step = 0
        context = self.context.format_map(config.__dict__)
        while current_step < self.tot_rounds:
            current_plan_str = (
                "Empty"
                if len(current_plan) == 0
                else "\n".join([f"{n}: {el}" for n, el in enumerate(current_plan)])
            )
            planner_prompt = f"""
            {context}
            You can do these and only these actions as steps:
            {"- \n".join(self.action_space)}
            You can use only steps written above!
            You will be given current non-finished plan.
            You should produce {self.tot_number} possible options for the next step of the plan. 
            Be precise. Think step by step.
            If you think, that plan is finished, print "End" as option.
            Your answer should be in the following form:
            {{
                thoughts: *your thoughts, think step by step*
                options: *{self.tot_number} options for the next step of the plan (should be a list of strings)*
            }}
            
            Current plan: 
            {current_plan_str}
            """
            result, _ = self.model.single_query_response(
                planner_prompt, structure=PlanStepFormat
            )
            result = self.model.get_structure_from_message(result)
            options = result.options
            options_str = "\n".join([f"{n}: {el}" for n, el in enumerate(options)])
            config.print_logs(f"Response: {result}")
            planner_tot_discriminator_prompt = f"""
            {context}
            You should choose the best option for the next step of the plan.
            Do not repeat plan steps and do not add redudant steps!
            If this step should be terminating (to finish the plan), set "terminate" to "True".
            Also set "terminate" to "True" if you chose "End" option.
            Your answer should be in the following form:
            {{
                thoughts: *your thoughts, think step by step*
                chosen_option: *best option*
                chosen_option_index: *index of the chosen option (should be a number)*
                is_terminating: *"True" or "False" (if plan should be finished with this step)*
            }}
            
            Current plan: 
            {current_plan_str}
            
            Possible options: 
            {options_str}
            """
            result, _ = self.model.single_query_response(
                planner_tot_discriminator_prompt, structure=PlanChoiceFormat
            )
            result = self.model.get_structure_from_message(result)
            config.print_logs(f"Response: {result}")
            next_option = options[result.chosen_option_index]
            current_plan.append(next_option)
            current_step += 1
            if result.is_terminating:
                break
        return (
            "Empty"
            if len(current_plan) == 0
            else "\n".join([f"{n}: {el}" for n, el in enumerate(current_plan)])
        )


class PlannerDomainReflector:
    def __init__(
        self,
        context,
        restriction,
        model,
        max_rounds,
        input_name,
        output_name,
        print_name=None,
    ):
        self.context = context
        self.restriction = restriction
        self.model = model
        self.max_rounds = max_rounds
        self.input_name = input_name
        self.output_name = output_name
        self.print_name = str(id(self)) if print_name is None else print_name

    def invoke(self, config):
        current_plan = config.get_by_name(self.input_name)
        context = self.context.format_map(config.__dict__)
        for i in range(self.max_rounds):
            planner_critic_cells_prompt = f"""
            {context}

            You will be given a prepared plan.
            Ensure that these requirements are met:
            {self.restriction}
            You can change this plan if any requirement is not met.

            Your answer should be in the following form:
            {{
                thoughts: *your thoughts, think step by step*
                changed: *"True" if you changed the plan or "False"*
                changes_list: *list of changes (should be list of strings)*
                plan: *modified or original plan*
            }}
            
            Plan:
            {current_plan}
            """
            result, _ = self.model.single_query_response(
                planner_critic_cells_prompt, structure=PlanReflector
            )
            result = self.model.get_structure_from_message(result)
            config.print_logs(f"Response: {result}")
            if not result.changed:
                break
            current_plan = result.plan
        config.set_by_name(self.output_name, result.plan)


class ReActExecutor:
    pass

class ToolCallingExecutor:
    def __init__(
        self,
        context,
        tools,
        model,
        output_name,
        plan=None,
        plan_type="proxy",
        state_model=None,  # None, "reflection", external
        post_hook=None,
        print_name=None,
    ):
        self.context = context
        self.model = model
        self.output_name = output_name
        self.plan = plan
        self.plan_type = plan_type
        self.state_model = state_model
        self.post_hook = post_hook
        self.print_name = str(id(self)) if print_name is None else print_name
        self.llm_tools = [model.get_tool_from_pydantic(tool) for tool, _ in tools]
        self.call_tools = {
            model.get_tool_from_pydantic(tool)["function"]["name"]: call
            for tool, call in tools
        }

    def invoke(self, config):
        config.print_header("ENTERING EXECUTOR")
        context = self.context.format_map(config.__dict__)
        config.print_logs(context)
        dialog = [
            {"role": "system", "content": context},
        ]
        while True:
            message, usage = self.model.single_query_response(
                dialog, tools=self.llm_tools
            )
            dialog.append(message)
            if message.tool_calls is None:
                break
            for toolcall in message.tool_calls:
                config.print_header("TOOL CALL")
                arguments = json.loads(toolcall.function.arguments)
                name = toolcall.function.name
                config.print_logs("Tool name: " + name)
                config.print_logs("Tool arguments: " + str(arguments))
                call_result = self.call_tools.get(name)(**arguments)
                config.print_logs("Tool executed: \n" + call_result)
                dialog.append(self.model.tool_feedback(call_result, toolcall.id))
        config.set_by_name(self.output_name, message.content)


class AgentState(TypedDict):
    messages: list[str]


class Agent:
    def __init__(self, graph, **kwargs):
        self.meta_config = kwargs
        self.config = ConfigProxy(kwargs)
        builder = StateGraph(AgentState)
        for node in graph:
            if isinstance(node, tuple):
                if len(node) == 2:
                    left = START if node[0] == "START" else node[0].print_name
                    right = END if node[1] == "END" else node[1].print_name
                    if right not in builder.nodes and right != END:
                        builder.add_node(
                            right, lambda _, node=node[1]: node.invoke(self.config)
                        )
                    builder.add_edge(left, right)
                if len(node) == 3:
                    left = START if node[0] == "START" else node[0].print_name
                    for value, el in node[2].items():
                        right = END if el == "END" else el.print_name
                        if right not in builder.nodes and right != END:
                            builder.add_node(
                                right, lambda _, node=el: node.invoke(self.config)
                            )
                    self.d = {value: END if el == "END" else el.print_name for value, el in node[2].items()}
                    builder.add_conditional_edges(
                        left,
                        lambda _, node=node[1]: self.config.get_by_name(node),
                        self.d,
                    )
                elif len(node) == 4:
                    left = START if node[0] == "START" else node[0].print_name
                    right_true = END if node[2] == "END" else node[2].print_name
                    right_false = END if node[3] == "END" else node[3].print_name

                    if right_true not in builder.nodes and right_true != END:
                        builder.add_node(
                            right_true, lambda _, node=node[2]: node.invoke(self.config)
                        )
                    if right_false not in builder.nodes and right_false != END:
                        builder.add_node(
                            right_false,
                            lambda _, node=node[3]: node.invoke(self.config),
                        )

                    builder.add_conditional_edges(
                        left,
                        lambda _, node=node[1]: self.config.get_by_name(node),
                        {True: right_true, False: right_false},
                    )
            if isinstance(node, list):
                for l, r in zip(node, node[1:]):
                    left = START if l == "START" else l.print_name
                    right = END if r == "END" else r.print_name
                    if right not in builder.nodes and right != END:
                        builder.add_node(
                            right, lambda _, node=r: node.invoke(self.config)
                        )
                    builder.add_edge(left, right)
        self.graph = builder.compile()

    def invoke(self, **kwargs):
        self.config.reset()
        self.config.update(self.meta_config)
        self.config.update(kwargs)
        state = {"messages": []}
        self.graph.invoke(state)
