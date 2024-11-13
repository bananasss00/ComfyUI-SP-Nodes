import { ComfyApp, app } from "../../../scripts/app.js";

app.registerExtension({
    name: "SP.Comfy.SlotDefaults.OptionalInputs",
    l_extension: null,
    n_suggestios: 0,

    init() {
        this.l_extension = app.extensions.find(ext => ext.name === 'Comfy.SlotDefaults');
        this.n_suggestios = app.ui.settings.getSettingValue('Comfy.NodeSuggestions.number');
        this.l_extension.beforeRegisterNodeDef = this.wrapBRND(this.l_extension.beforeRegisterNodeDef);
    },

    wrapBRND(originalFunction) {
        return function(...args) {
            const nodeData = args[1]; 
            const modifiedNodeData = JSON.parse(JSON.stringify(nodeData));
            
            // merge required+optional inputs for Comfy.SlotDefaults
            const required = Object.values(modifiedNodeData.input?.required || {});
            const optional = Object.values(modifiedNodeData.input?.optional || {});
            modifiedNodeData.input.required = [...required, ...optional];

            args[1] = modifiedNodeData;

            const result = originalFunction.apply(this, args);
            return result;
        };
    }
  });

  