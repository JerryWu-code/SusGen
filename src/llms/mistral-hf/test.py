# Author: "WHATX" -- Wu Qilong
# Institute: National University of Singapore, A Star IHPC
# Description: Use this to:
#    1. Load the model and tokenizer from the local directory and test inference.
#    2. Turn the weights into consolidated format for deployment.

#############################################################################
from template import load_model, generate_text, instr_prompt
import warnings
warnings.filterwarnings("ignore")

def test():
    model_path =  "../../../ckpts/Mistral-7B-Instruct-v0.2-hf"
    model, tokenizer, device, _ = load_model(model_path)

    # Case 1
    prompt1 = instr_prompt("What is tcfd format in the context of climate change?")
    # Case 2
    user_instruction = (
        "You are a senior equity analyst with expertise in climate science "
        "evaluating a company 's sustainability report, "
        "Please concentrate on the organizations governance around climate-related risks and opportunities."
    )
    question = ("Text: Our Board of Directors is responsible for all sustainability matters at Wolfspeed, including climate change, through our Governance and Nominations Committee. Further information about our Board’s oversight of sustainability can be found in the Sustainability Oversight section of this report.\n Our CEO, who is also the Company’s President and a Board member, is also responsible for climate-related issues impacting the company because he has oversight of departments within Wolfspeed, including those that manage climate-related issues (e.g., environment, health and safety, sustainability, emergency management, product development, operations, etc.). More information about our CEO’s role with the Board of Directors can be found in the Board of Directors and Committee Composition section of this report and on our Board of Directors web page.\n Sustainability-related information is presented to our Board of Directors at least once per year, which covers a range of topics, including environmental performance (GHG emissions/climate change, water, etc.) and social responsibility efforts.\n Our Board of Directors also discusses climate change risks as important matters arise because our manufacturing facilities are not located in areas that are typically directly impacted by climate- related events (e.g., tropical storms, droughts, etc.). Indirectly, our Board discusses climate-related opportunities offen, as our business, and more speciffcally our products, are designed to reduce energy usage, and therefore greenhouse gas emissions, which directly affect climate change. For example, our Board helps guide our business strategy, part of which focuses on the development of Silicon Carbide products that enable auto manufacturers to reach their goals of electric vehicle production and adoption around the world. Our Board of Directors also help guide our Sustainability strategy, including goals and targets development. Refer to the Sustainability Goals section of this report for more information about our current sustainability goals and targets. Our climate-related risks and opportunities over the short-, medium-, and long-term can be found in the Our Climate Change Risks and Our Climate Change Opportunities subsection. The risks and opportunities reported here refer to those that could have a substantive ffnancial or strategic impact to our business.\n We define a substantive ffnancial or strategic impact as something that will cause signiffcant impact to our business both internally (i.e., our direct operations) or externally (i.e., our upstream and downstream value chain). We use $1million USD to establish a threshold for substantive ffnancial impact when determining potential impacts due to climate change.\n Our short-term horizon was chosen to be 0-1 years because our budgets are currently established on a shorterterm timeframe. Our medium-term horizon was chosen to be 1-10 years based on our anticipated timeline for our current capacity expansion efforts that are planned to be completed in 2024. Our long-term horizon of 10-100 years is not currently aligned with other business practice time horizons."
                "Describe the organization's governance around climate-related risks and opportunities"
    )
    prompt2 = f"{user_instruction}\n Question: {question}"

    # for test
    text = "CPP Investments is in the process of identifying and monitoring climate-related factors that may impact our investment portfolio, with an emphasis on managing risk as central to our legislative objectives. This includes a broad array of complex interrelated risks such as physical and transition risks, for which there is no established historical fact set. Our Risk Group uses various approaches including scenario analysis and 'bow-tie' risk & control assessments to assess climate change risk."
    p3 = instr_prompt(
        "Process the text following the instructions below:\n"
        "1.Replace all the specific company entity name with \"we\" or \"our company\""
        "2.Replace other private information with generic terms\n"
        f"\nText:\n{text}")

    # Define args
    args = {
        "max_length": 8096,
        "do_sample": True,
        "temperature": 0.2,
        "top_p": 0.9,
        "top_k": 40,
        "num_return_sequences": 1
    }
    _, answer = generate_text(model, tokenizer, device, prompt=p3, args=args)

def main():
    test()

if __name__ == "__main__":
    main()