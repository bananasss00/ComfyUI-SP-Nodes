import { ComfyApp, app } from "../../../scripts/app.js";
import { ComfyDialog, $el } from "../../../scripts/ui.js";
import { api } from "../../../scripts/api.js";

let wildcards_load_promise = null;
let wildcards_list = [];
function load_wildcards() {
	if (!wildcards_load_promise) {
		wildcards_load_promise = (async () => {
			try {
				let res = await api.fetchApi('/prompt_checker/wildcards/list');
				let data = await res.json();
				wildcards_list = data.data;
			} catch (e) {
				console.error("Failed to load wildcards list:", e);
				wildcards_load_promise = null;
			}
		})();
	}
	return wildcards_load_promise;
}

app.registerExtension({
	name: "comfy.sp_nodes.prompt_checker.impactpack_wildcards",

	// Runs at registration time (before widgets are built), so the combo
	// widget picks up the list fetched from the server instead of the
	// placeholder value baked into INPUT_TYPES.
	async beforeRegisterNodeDef(nodeType, nodeData) {
		if (nodeData.name != 'PromptChecker') return;
		await load_wildcards();
		const values = wildcards_list.length ? wildcards_list : ["Select the Wildcard to add to the text"];
		const required = nodeData.input?.required ?? (nodeData.input.required = {});
		required["Select to add Wildcard"] = [values,];
	},

	nodeCreated(node) {
		if (node.comfyClass != "PromptChecker") return;
		node._wvalue = "Select the Wildcard to add to the text";
		const combo_widget = node.widgets.find((w) => w.name == 'Select to add Wildcard');
		const wildcard_text_widget = node.widgets.find((w) => w.name == 'prompt');

		Object.defineProperty(combo_widget, "value", {
			set: (value) => {
				if (value != node._wvalue) {
					if (wildcard_text_widget.value != '')
						wildcard_text_widget.value += ', ';
					wildcard_text_widget.value += value;
				}
			},
			get: () => { return node._wvalue; }
		});

		Object.defineProperty(combo_widget.options, "values", {
			set: (x) => {},
			get: () => { return wildcards_list; }
		});

		// Preventing validation errors from occurring in any situation.
		combo_widget.serializeValue = () => { return node._wvalue; }
	}
});
