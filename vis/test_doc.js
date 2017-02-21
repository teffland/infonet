var doc = {
  "tokens":["This", "is", "a", "test", "relation", "with", "a", "few", "extra", "words"],
  "annotations": [
    {
      "ann-type": "node",
      "ann-uid": "m_1",
      "ann-span": [0,1],
      "node-type": "entity",
      "type": "OBJ"
    },
    {
      "ann-type": "node",
      "ann-uid": "m_2",
      "ann-span": [3,5],
      "node-type": "entity",
      "type": "OBJ"
    },
    {
      "ann-type": "edge",
      "ann-uid": "r_1",
      "ann-left": "m_1",
      "ann-right": "m_2",
      "edge-type": "coref",
      "type": "--SameAs-->"
    },
  ]
};

var container = {
  'w':100,
  'h':500,
  'pad':{'x':20, 'y':20},
  'border':{'w':1, 'color':'black'}
};

// container
var svg = d3.select('div.doc')
            .append('svg')
            .attr("width", container.w + 2*container.pad.x)
            .attr("height", container.h + 2*container.pad.y)
            .append("g")
            .attr("transform", "translate("+container.pad.x+","+container.pad.y+")");

var borderBox = svg.append("rect")
                .attr("x", 0)
                .attr("y", 0)
       			    .attr("width", container.w)
                .attr("height", container.h)
       		      .style("stroke", container.border.color)
       		      .style("fill", "none")
       		      .style("stroke-width", container.border.w);

var textbox = svg.append("g")
                 .classed("text", true);

// get mapping from token start index to mention
var start2mention {}
for (var i=0; i < doc.annotations.length; i++) {
  var ann = doc.annotations[i];
  if (ann['ann-type'] == 'node') {
    if (ann['ann-span'][0] in start2mention) {
      start2mention[ann['ann-span'][0]].push(ann);
    } else {
      start2mention[ann['ann-span'][0]] = [ann];
    }
}

function createLine(tokens, s2m, line_num,
                    line_height=15, padx=5, pady=5, space=3) {
  // create line container
  var y = line_num * (line_height+pady)+pady;
  var line = textbox.append("g")
                    .attr("transform", "translate("+padx+","+y+")");
  // create text subline
  var textline = line.append("svg")
                     .attr("width", container.w - 2*padx)
                     .attr("height", line_height + pady);
  var x=0;
  for (var i=0, len=tokens.length; i < len; i++) {
    var token = tokens.shift();
    console.log(token);
    var text = textline.append("text")
                       .attr("x", x)
                       .attr("y", line_height+pady)
                       .text(token)
                       .classed("token", true);
    token_width = text.node().getBBox().width;
    x += space + token_width;
    // this token put us over the line width
    if ( x > container.w - padx ) {
      if (i == 0 && token_width > container.w-padx) return;
      text.remove();
      tokens.unshift(token);
      return;
    }
  }
}

function createLines(tokens, start2mention) {
  for (var i=0; i<10; i++) {
    // console.log('i', i);
    createLine(tokens, start2mention, i);
  }
}
createLines(doc.tokens);



// function createLines(tokens) {
//   var lines = [];
//   var pad = {'x':5, 'y':10};
//   var s = 4
//       x = 0
//       y = 15
//       line_i = 0;
//   var g = svg.append("g")
//     .attr("transform", "translate("+pad.x+",0)");
//   for  (var i=0, len=tokens.length; i < len; i++) {
//     var token = tokens[i];
//     console.log(token, x, y);
//     var text = g.append("text")
//      .attr("x", x)
//      .attr("y", y)
//      .text(token)
//      .classed("token", true);
//     x = x + s + text.node().getBBox().width;
//     if ( x > container.w-pad.x ) {
//       line_i += 1;
//       g = svg.append("g")
//         .attr("transform", "translate("+pad.x+","+line_i*(y+pad.y)+")");
//       // console.log(text.remove());
//       text.remove();
//       x = 0;
//       var text = g.append("text")
//        .attr("x", x)
//        .attr("y", y)
//        .text(token)
//        .classed("token", true);
//       x = x + s + text.node().getBBox().width;
//     }
//   }
// }
// createLines(doc.tokens);
