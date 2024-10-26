import { ComfyApp, app } from "../../../scripts/app.js";
import { ComfyDialog, $el } from "../../../scripts/ui.js";
import { api } from "../../../scripts/api.js";

let missing_nodes = {};
async function load_missing_nodes() {
	let res = await api.fetchApi('/sp_group_nodes/missing_nodes');
	let data = await res.json();
	missing_nodes = data.data;
}

load_missing_nodes();


const requirements = {
    SP_Supir: ["ComfyUI-SUPIR", "ComfyUI_essentials"],
    SP_SDKSampler: ["rgthree-comfy", "ComfyUI-Impact-Pack"],
    SP_FluxKSampler: ["rgthree-comfy", "ComfyUI-Impact-Pack"],
    SP_FluxLoader: ["ComfyUI_bitsandbytes_NF4-Lora", "ComfyUI-GGUF"],
  };

function checkAndSuggestExtensions(nodeClass, missingNodes) {
    const requiredExtensions = requirements[nodeClass];

    if (!requiredExtensions) return;

    const missingExtensions = [];

    requiredExtensions.forEach(extensionName => {
        if (missingNodes[extensionName]) {
            const installUrl = missingNodes[extensionName].install_url;
            missingExtensions.push(`- ${extensionName}: ${installUrl}`);
        }
    });

    if (missingExtensions.length > 0) {
        alert(`The node ${nodeClass} requires the following extensions:\n${missingExtensions.join('\n')}`);
    }
}

app.registerExtension({
	name: "comfy.sp_nodes.group_nodes",

	nodeCreated(node, app) {
        if (!node.comfyClass.startsWith("SP_"))
            return;
        
        console.log('check', node.comfyClass);
        checkAndSuggestExtensions(node.comfyClass, missing_nodes);
	}
});
