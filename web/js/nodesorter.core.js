import { app } from '../../../scripts/app.js'

app.registerExtension({
    name: "NodeSorter.Core",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        
        LGraphCanvas.onMenuAdd = function (node, options, e, prev_menu, callback) {

            var canvas = LGraphCanvas.active_canvas;
            var ref_window = canvas.getCanvasWindow();
            var graph = canvas.graph;
            if (!graph)
                return;

            function inner_onMenuAdded(base_category ,prev_menu){
        
                var categories  = LiteGraph.getNodeTypesCategories(canvas.filter || graph.filter).filter(function(category){return category.startsWith(base_category)});
                var entries = [];
        
                categories.map(function(category){
        
                    if (!category) 
                        return;
        
                    var base_category_regex = new RegExp('^(' + base_category + ')');
                    var category_name = category.replace(base_category_regex,"").split('/')[0];
                    var category_path = base_category  === '' ? category_name + '/' : base_category + category_name + '/';
        
                    var name = category_name;
                    if(name.indexOf("::") != -1) //in case it has a namespace like "shader::math/rand" it hides the namespace
                        name = name.split("::")[1];
                            
                    var index = entries.findIndex(function(entry){return entry.value === category_path});
                    if (index === -1) {
                        entries.push({ value: category_path, content: name, has_submenu: true, callback : function(value, event, mouseEvent, contextMenu){
                            inner_onMenuAdded(value.value, contextMenu)
                        }});
                    }
                    
                });
        
                var nodes = LiteGraph.getNodeTypesInCategory(base_category.slice(0, -1), canvas.filter || graph.filter );
                nodes.map(function(node){
        
                    if (node.skip_list)
                        return;
        
                    var entry = { value: node.type, content: node.title, has_submenu: false , callback : function(value, event, mouseEvent, contextMenu){
                        
                            var first_event = contextMenu.getFirstEvent();
                            canvas.graph.beforeChange();
                            var node = LiteGraph.createNode(value.value);
                            if (node) {
                                node.pos = canvas.convertEventToCanvasOffset(first_event);
                                canvas.graph.add(node);
                            }
                            if(callback)
                                callback(node);
                            canvas.graph.afterChange();
                        
                        }
                    }
        
                    entries.push(entry);
        
                });
                
                entries.sort(function(a, b) {
                    return a.content.localeCompare(b.content);
                });
            
                new LiteGraph.ContextMenu( entries, { event: e, parentMenu: prev_menu }, ref_window );
        
            }
        
            inner_onMenuAdded('',prev_menu);
            return false;
        
        };
        
        // console.log('patched');
    }
});
