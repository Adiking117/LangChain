# main.py
from agent_tools import build_react_agent

if __name__ == "__main__":
    agent, pdf_agent = build_react_agent()
    print("ReAct PDF Agent Ready.")
    print("Commands: load_pdf <path>, summarize <query>, ask <question>, quiz <topic>, or free text for agent reasoning.")
    
    while True:
        cmd = input(">> ").strip()
        if cmd.lower() in ["exit", "quit"]:
            break
        try:
            if cmd.startswith("quiz"):
                topic = cmd[len("quiz "):] if len(cmd) > 5 else None
                print(pdf_agent.generate_quiz(topic)) # type: ignore
            else:
                print(agent.run(cmd))
        except Exception as e:
            print(f"Error: {e}")
