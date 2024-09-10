import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";
import { $el } from "../../../scripts/ui.js";

async function lora_from_folder_list(lora_folder) {
	const response = await api.fetchApi('/sp_nodes/lora_from_folder_list', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            lora_folder: lora_folder,
        })
    });
    let data = await response.json();
    // console.log('data:');
    // console.log(data.data);
	return data.data;
}

app.registerExtension({
    name: 'comfy.sp_nodes.lora_loader',

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if(!nodeData?.category?.startsWith("SP-Nodes")) {
			return;
		  }
        
		switch (nodeData.name) {
			case "LoraLoaderFromFolder":
                const originalOnNodeCreated = nodeType.prototype.onNodeCreated || function() {};

				nodeType.prototype.onNodeCreated = function () {
                    originalOnNodeCreated.apply(this, arguments);

                    this.addWidget("button", "Update", null, () => {
                        this._prevPath = '';
                    });
				}
				break;
		}	
		
	},

    nodeCreated(node, app) {
		if(node.comfyClass == "LoraLoaderFromFolder") {
			node._prevPath = "";

			var tbox_id = 0;
			var combo_id = 1;
            // console.log(node.widgets);
            
            let loraValues = [];

            async function fetchValuesAsync() {
                const path = node.widgets[tbox_id].value;

                if (node._prevPath === path)
                    return;
                
                node._prevPath = path;
                
                let lora_folder = node.widgets[tbox_id].value;
                let loras = await lora_from_folder_list(lora_folder);
                // console.log(`lora_folder: ${lora_folder}, loras: ${loras}`);
                loraValues = loras;
                
                console.log('LoraLoaderFromFolder updated');
                
                if (loras.length === 0) {
                    node.widgets[combo_id].value = '';
                }
            }

			Object.defineProperty(node.widgets[combo_id].options, "values", {
				set(x) {},
				get() {
                    fetchValuesAsync();
					return loraValues;
				}
			});
        }
    }
})
