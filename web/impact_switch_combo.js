import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

/**
 * Adds a custom menu handler to a node type.
 * The handler extends the node's extra menu options.
 *
 * @param {Function} nodeType - The node type constructor.
 * @param {Function} callback - The callback to modify the extra menu options.
 */
function addMenuHandler(nodeType, callback) {
  const originalGetExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
  nodeType.prototype.getExtraMenuOptions = function (...args) {
    const options = originalGetExtraMenuOptions.apply(this, args);
    callback.apply(this, args);
    return options;
  };
}

/**
 * Creates a new node and positions it relative to a reference node.
 *
 * @param {string} name - The name of the node type to create.
 * @param {Object} referenceNode - The node next to which the new node is positioned.
 * @param {Object} [options] - Options for positioning and selection.
 * @param {boolean} [options.select=true] - Whether to select the node after creation.
 * @param {number} [options.shiftY=0] - Vertical offset from the reference node.
 * @param {boolean} [options.before=false] - If true, positions the new node to the left of the reference node.
 * @returns {Object} The created node.
 */
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

app.registerExtension({
  name: "SP_ImpactSwitchCombo",

  registerCustomNodes() {
    class SP_ImpactSwitchCombo extends LGraphNode {
      // Enable widget serialization.
      serialize_widgets = true;
      // Set the canvas reference.
      canvas = app.canvas;
      comfyClass = "SP_ImpactSwitchCombo";

      constructor(title) {
        super(title);
        // Ensure the properties object exists.
        this.properties = this.properties || { select: "" };

        // Add a combo widget for "select"
        this.addWidget(
          "combo",
          "select",
          this.properties.select,
          (value) => {
            console.log("Combo value changed:", value);
            this.properties.select = value;

            // Update the 'select' widget of all connected nodes.
            const outputLinks = this.outputs[0].links;
            outputLinks.forEach((linkId) => {
              const link = app.graph.links[linkId];
              const connectedNode = app.graph.getNodeById(link.target_id);
              const selectWidget = connectedNode.widgets.find(
                (widget) => widget.name === "select"
              );
              const valueIndex = this.widgets[0].options.values.indexOf(value);
              if (selectWidget) {
                // Set the connected widget value (1-based index)
                selectWidget.value = valueIndex + 1;
              }
            });

            this.serialize();
          },
          { values: [] }
        );

        // Add an output slot named "select" of type INT.
        this.addOutput("select", "INT");

        // Handle connection changes for this node.
        this.onConnectionsChange = (slotType, slot, isChangeConnect, linkInfo) => {
          // Proceed only if an output connection is made/changed.
          if (slotType !== 2 || !isChangeConnect || !linkInfo) {
            this.widgets[0].options.values = [];
            return;
          }

          const connectedNode = app.graph.getNodeById(linkInfo.target_id);
          // Disconnect if the connected node is not of type "ImpactSwitch".
          if (!connectedNode || connectedNode.type !== "ImpactSwitch") {
            if (connectedNode) {
              this.disconnectOutput(0);
            }
            return;
          }

          // Get input slots from the connected ImpactSwitch node that start with "input".
          const filteredInputs = connectedNode.inputs.filter((input) =>
            input.name.startsWith("input")
          );
          // Remove the last input if at least one exists.
          if (filteredInputs.length > 0) {
            filteredInputs.pop();
          }
          console.log("Filtered inputs:", filteredInputs);

          // Use the labels (or names) of the filtered inputs as combo options.
          const inputLabels = filteredInputs.map((input) => input.label || input.name);
          const selectWidget = connectedNode.widgets.find((widget) => widget.name === "select");

          // Update the combo widget options and set the current value.
          this.widgets[0].options.values = inputLabels;
          this.widgets[0].value = inputLabels[selectWidget.value - 1];

          // Mark this node as virtual so it won’t be serialized as part of the final prompt.
          this.isVirtualNode = true;
        };
      }
    }

    // Register the custom node type with LiteGraph.
    LiteGraph.registerNodeType(
      "SP_ImpactSwitchCombo",
      Object.assign(SP_ImpactSwitchCombo, {
        title: "SP_ImpactSwitchCombo",
      })
    );
    SP_ImpactSwitchCombo.category = "SP-Nodes";
  },

  /**
   * Before registering a node definition, add a custom menu entry to ImpactSwitch nodes.
   *
   * @param {Function} nodeType - The node type constructor.
   * @param {Object} nodeData - Data for the node being registered.
   */
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name === "ImpactSwitch") {
      addMenuHandler(nodeType, function (_, options) {
        options.unshift({
          content: "Add Switch Combo",
          callback: () => {
            // Create a new SP_ImpactSwitchCombo node positioned before the current ImpactSwitch node.
            const comboNode = addNode("SP_ImpactSwitchCombo", this, { before: true });
            let slot = this.findInputSlot("select");
            // If no input slot exists for "select", convert the widget to an input.
            if (slot === -1) {
              this.convertWidgetToInput(this.widgets.find((w) => w.name === "select"));
              slot = this.findInputSlot("select");
            }
            comboNode.connect(0, this, slot);
          },
        });
      });
    }
  },
});
