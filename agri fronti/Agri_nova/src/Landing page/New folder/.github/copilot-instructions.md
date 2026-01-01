- [x] Verify that the copilot-instructions.md file in the .github directory is created.

- [x] Clarify Project Requirements
	Ask for project type, language, and frameworks if not specified. Skip if already provided.

- [x] Scaffold the Project
	Ensure that the previous step has been marked as completed. Call the project setup tool with the chosen template, run scaffolding in the current directory, and fall back to manual creation if scaffolding fails.

- [x] Customize the Project
	Develop a plan, then apply modifications according to user requirements.

- [x] Install Required Extensions
	Install only the extensions mentioned in get_project_setup_info. If none are required, mark the step complete.

- [x] Compile the Project
	Install any missing dependencies, run diagnostics, and resolve build issues.

- [x] Create and Run Task
	If the project benefits from a task, add it through the create_and_run_task tool and execute it once.

- [x] Launch the Project
	Run the application (dev server or equivalent) after confirming with the user.

- [x] Ensure Documentation is Complete
	Keep README.md and this checklist up to date. Remove obsolete guidance once tasks are done.

## Execution Guidelines
**Progress Tracking**
- Use manage_todo_list to reflect each step.
- Update the status before starting and after finishing a step.

**Communication Rules**
- Avoid verbose explanations or dumping command output.
- If a step is skipped, note it briefly.
- Discuss structure only when asked.

**Development Rules**
- Work from the current directory unless told otherwise.
- Avoid external media unless explicitly requested.
- Use placeholders only when clearly labeled.

**Folder Creation Rules**
- Stay in the current root unless the user requests a new folder.
- Create `.vscode` only when tasks/debug configs are required.

**Extension Installation Rules**
- Install only what get_project_setup_info prescribes.

**Project Content Rules**
- Assume “Hello World” defaults if requirements are unclear.
- Skip unnecessary links or integrations.
- Prompt for clarification before adding unconfirmed features.

**Task Completion Rules**
- Finish when the project scaffolds, compiles, has updated README + instructions, and the user knows how to launch/debug it.

- Work through each checklist item systematically.
- Keep communication concise and focused.
- Follow development best practices.
