import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

class DynamicCombo {
    constructor(node) {

        this.node = node;
        this.node.properties = { name: 'combo', values: '1;2;3;4;5;6', separator: ';' };
        this.node.size = [210,LiteGraph.NODE_SLOT_HEIGHT*3.4];
        this.node.widgets[0].hidden = true;
        this.node.widgets[0].type = "hidden";
        this.node.outputs[0].label = this.node.properties.name;

        this.selectedWidget = this.node.widgets[0];

        const self = this;
        const w = {
            type: 'combo', //.toLowerCase()
            get name() { return self.node.properties.name },
            get value() {
                return self.selectedWidget.value;
            },
            set value(v) { 
                // work for comfyui frontend < v1.9.8
                self.selectedWidget.value = v
            },
            callback: (v) => {
                // fix for comfyui frontend >= v1.9.8
                self.selectedWidget.value = v;
            },
            options: {
                serialize: false,
                get values() {
                    return self.node.properties.values.split(self.node.properties.separator); 
                }
              }};
        this.node.widgets.splice(0, 0, w);

        this.node.onAdded = function ()
        {
            
        };

        this.node.onPropertyChanged = function (name, value)
        {
            if (name === 'name') {
                self.node.outputs[0].label = value;
            }
        }

        this.node.valueUpdate = function(e)
        {
            console.log(e)
        }

        this.node.computeSize = function()
        {
            return [210,LiteGraph.NODE_SLOT_HEIGHT*3.4];
        }
    }
}

app.registerExtension(
{
    name: "SP_DynamicCombo",
    async beforeRegisterNodeDef(nodeType, nodeData, _app)
    {
        if (nodeData.name === "SP_DynamicCombo")
        {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function ()
            {
                if (onNodeCreated) onNodeCreated.apply(this, []);
                this.dynamicCombo = new DynamicCombo(this);
            }
        }
    }
});
