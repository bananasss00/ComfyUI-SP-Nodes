import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "SP.Pipes.SyncRename",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // Only apply logic to nodes starting with SP_AnyPipe
        if (nodeData.name.startsWith("SP_AnyPipe")) {

            /**
             * Applies the synchronization logic to a specific node.
             * This function attaches getters/setters to input and output labels.
             */
            const applySyncLogic = (node) => {
                
                // Helper to watch a specific slot
                function watchLabel(slotArray, index, isInput) {
                    const slot = slotArray[index];
                    if (!slot) return;

                    // Check if we have already attached the watcher to this specific slot object.
                    // If we don't check this, loading a workflow might crash or create double hooks.
                    if (slot._sp_sync_attached) return;

                    // Initialize with the current label or name
                    let _val = slot.label || slot.name;

                    // Mark this slot as "watched" so we don't re-apply logic
                    slot._sp_sync_attached = true;

                    // Redefine the 'label' property
                    Object.defineProperty(slot, "label", {
                        get() { return _val; },
                        set(newVal) {
                            // If value is unchanged, do nothing
                            if (_val === newVal) return;
                            
                            _val = newVal;
                            
                            // Find the mirror slot on the opposite side
                            const otherArray = isInput ? node.outputs : node.inputs;
                            
                            if (otherArray && otherArray[index]) {
                                const otherSlot = otherArray[index];
                                const currentOtherName = otherSlot.label || otherSlot.name;
                                
                                // Update the other slot if the name differs
                                if (currentOtherName !== newVal) {
                                    otherSlot.label = newVal;
                                    // If the other slot is also watched, its setter will trigger,
                                    // but the equality check above prevents infinite loops.
                                }
                            }
                        },
                        configurable: true
                    });
                }

                // Apply to all inputs
                if (node.inputs) {
                    for (let i = 0; i < node.inputs.length; i++) {
                        watchLabel(node.inputs, i, true);
                    }
                }

                // Apply to all outputs
                if (node.outputs) {
                    for (let i = 0; i < node.outputs.length; i++) {
                        watchLabel(node.outputs, i, false);
                    }
                }
            };

            // ----------------------------------------------------------------
            // HOOK 1: onNodeCreated
            // Triggered when adding a new node to the graph manually.
            // ----------------------------------------------------------------
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                applySyncLogic(this);
                return r;
            };

            // ----------------------------------------------------------------
            // HOOK 2: onConfigure
            // Triggered when loading a workflow (JSON) or initializing a Subgraph.
            // This ensures the watchers are re-applied to the restored slots.
            // ----------------------------------------------------------------
            const onConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function () {
                const r = onConfigure ? onConfigure.apply(this, arguments) : undefined;
                // We apply logic AFTER configuration to capture the loaded slots/labels
                applySyncLogic(this);
                return r;
            };
        }
    }
});