import { ComfyApp, app } from "../../../scripts/app.js";
import { ComfyDialog, $el } from "../../../scripts/ui.js";
import { api } from "../../../scripts/api.js";

let wildcards_list = [];
async function load_wildcards() {
	let res = await api.fetchApi('/prompt_checker/wildcards/list');
	let data = await res.json();
	wildcards_list = data.data;
}

load_wildcards();

app.registerExtension({
	name: "comfy.sp_nodes.prompt_checker.impactpack_wildcards",

	nodeCreated(node, app) {
		if(node.comfyClass == "PromptChecker") {
			node._wvalue = "Select the Wildcard to add to the text";
			const combo_widget = node.widgets.find((w) => w.name == 'Select to add Wildcard');
			const wildcard_text_widget = node.widgets.find((w) => w.name == 'prompt');

			Object.defineProperty(combo_widget, "value", {
				set: (value) => {
				        // const stackTrace = new Error().stack;
                        // if(stackTrace.includes('inner_value_change')) {
                            if(value !== "Select the Wildcard to add to the text") {
                                if(wildcard_text_widget.value != '')
                                    wildcard_text_widget.value += ', '

	                            wildcard_text_widget.value += value;
                            }
                        // }
					},
				get: () => { return "Select the Wildcard to add to the text"; }
			});
			Object.defineProperty(combo_widget.options, "values", {
			    set: (x) => {},
			    get: () => {
			    	return wildcards_list;
			    }
			});

			// Preventing validation errors from occurring in any situation.
			combo_widget.serializeValue = () => { return "Select the Wildcard to add to the text"; }
		}
	}
});
