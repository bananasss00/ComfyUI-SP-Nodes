import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";


function addMenuHandler(nodeType, callback) {
    const originalGetExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
    nodeType.prototype.getExtraMenuOptions = function (...args) {
        const options = originalGetExtraMenuOptions.apply(this, args);
        callback.apply(this, args);
        return options;
    };
}


function addNode(name, referenceNode, options = {}) {
    const { select = true, shiftY = 0, before = false } = options;
    const node = LiteGraph.createNode(name);
    app.graph.add(node);
    node.pos = [
        before
        ? referenceNode.pos[0] - node.size[0] - 30
        : referenceNode.pos[0] + referenceNode.size[0] + 30,
        referenceNode.pos[1] + shiftY,
    ];
    if (select) {
        app.canvas.selectNode(node, false);
    }
    return node;
}
  
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

function showSubMenu(value, options, e, menu, node) {
    const behaviorOptions = [];

    for (let widget of node.widgets) {
        if (widget.type !== "combo") continue;

        behaviorOptions.push({
            content: widget.name,
            callback: () => {
                const values = widget.options.values.join('\n');

                // Create a new SP_ImpactSwitchCombo node positioned before the current ImpactSwitch node.
                const comboNode = addNode("SP_DynamicCombo", node, { before: true });
                let slot = node.findInputSlot(widget.name);
                // If no input slot exists for "select", convert the widget to an input.
                if (slot === -1) {
                    node.convertWidgetToInput(node.widgets.find((w) => w.name === widget.name));
                    slot = node.findInputSlot(widget.name);
                }

                // Connect the new node to the input slot.
                comboNode.connect(0, node, slot);

                // Set the name and values of the new node.
                if (widget.options.values.length > 0) {
                    comboNode.widgets[0].value = widget.options.values[0];
                }
                comboNode.outputs[0].label = widget.name;
                comboNode.properties.name = widget.name;
                comboNode.properties.values = values;
                comboNode.properties.separator = '\n';
            }
        })
    }

    new LiteGraph.ContextMenu(behaviorOptions, {
        event: e,
        callback: null,
        parentMenu: menu,
        node: node
    });

    return false;  // This ensures the original context menu doesn't proceed
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

        addMenuHandler(nodeType, function (_, options) {
            options.unshift({
                content: "Extract Dynamic Combo",
                has_submenu: true,
                callback: showSubMenu
            })
        })
    }
});
