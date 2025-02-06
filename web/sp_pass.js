import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

let type_defs = new Set();

const originalRegisterNodesFromDefs = app.registerNodesFromDefs;
app.registerNodesFromDefs = async function(defs) {
  for (let key in defs) {
    for (let idx in defs[key].output)
      if (!Array.isArray(defs[key].output[idx]))
        type_defs.add(defs[key].output[idx]);
  }

  type_defs = [...type_defs.values()];

  const result = await originalRegisterNodesFromDefs.call(this, defs);
  return result;
};


function reconnectOutputs(node) {
  // 0246.reroute node
  // reconnect outputs for use everywhere
  const currOutputNodes = node.getOutputNodes(0) || [];
  const prevOutputSlots = (node.outputs[0].links || []).map(
    linkId => app.graph.links[linkId].target_slot
  );

  node.disconnectOutput(0);

  currOutputNodes.forEach((outputNode, index) => {
    node.connect(0, outputNode, prevOutputSlots[index]);
  });
}

function setType(node, type, change_title) {
  node.inputs[0].type = node.outputs[0].type = type;
  node.inputs[0].label = node.outputs[0].label = type?.toLowerCase();
  if (change_title) {
    node.title = type;
  }
}

// Register extensions
app.registerExtension({
  name: "Comfy.SP_Pass",
  
  async beforeRegisterNodeDef(nodeType, nodeData, app) {
    if (nodeData.name === "SP_Pass") {
      // Node Created
      const onNodeCreated = nodeType.prototype.onNodeCreated;
      nodeType.prototype.onNodeCreated = async function () {
        const ret = onNodeCreated
          ? onNodeCreated.apply(this, arguments)
          : undefined;

        const node_title = await this.getTitle();
        
        setType(this, this.widgets[0].value, false);
        reconnectOutputs(this);

        // assign type values
        this.widgets[0].callback = function(value, widget, node) {
            setType(node, value, true);
            reconnectOutputs(node);
        };
        this.widgets[0].options.values = type_defs;

        return ret;
      };
    }
  },
});
