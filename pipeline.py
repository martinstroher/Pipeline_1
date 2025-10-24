from ner_term_extractor.1.1 - ner_term_extractor import run_step_1
from term_aggregator_for_ner_model_output import run_step_2
from nld_generator import run_step_3
from term_categorizer import run_step_4

def main():
    run_step_1()
    run_step_2()
    run_step_3()
    run_step_4()

if __name__ == "__main__":
    main()
