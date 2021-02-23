// CitationGraphs.go project CitationGraphs.go
package CitationGraphs

import (
	"bufio"
	"fmt"
	"log"
	"os"
	"regexp"
	"strconv"
	"strings"
)

var reUnicodeHex *regexp.Regexp
var reUnicodeDec *regexp.Regexp

func init() {
	reUnicodeHex = regexp.MustCompile("&#[Xx]([A-Fa-f0-9])+;")
	reUnicodeDec = regexp.MustCompile("&#([0-9])+;")
}

/*
# ===========================================================================================
# struct CitationNode
# brief description: The node structure of a citation graph, the data of which are collected
#                    from crawling the website https://academic.microsoft.com , which is the
#                    public service website of MAG(Microsoft Academic Graph)
# fields:
#   id: The MAG(Microsoft Academic Graph) id of the paper at this node.
#       We can access the detail of the paper with the id by navigating the web link:
#       https://academic.microsoft.com/paper/$id
#   year: The year of publication of this paper.
#   title: The title of the paper.
#   labels: The labels from MAG(Microsoft Academic Graph).
#   refs: The references of this paper collected from MAG.
#         Please note that this field could be inaccurate: MAG suffers delay of info.
#         Many new papers with refs don't have refs listed in MAG.
#   cites: The citations to this paper from other papers (also collected from MAG).
#         Please note that this field is as inaccurate as refs due to the same reason.
*/
type CitationNode struct {
	ID     int64
	Year   int64
	Title  string
	Labels []string
	Refs   []int64
	Cites  []int64
}

/*
# ===========================================================================================
# struct CitationGraph
# brief description: A data structure of citation graph, the data of which are collected from
#                    the website https://academic.microsoft.com , which is the public service
#                    website of MAG(Microsoft Academic Graph)
# fields:
#   nodes: The node of struct CitationNode stored in a dictionary with key = id
#   toBeAnalyzed: The list of nodes to be analyzed. In ijcai_clustering dataset, these nodes
#                 are those published in IJCAI.
*/
type CitationGraph struct {
	Nodes        map[int64]*CitationNode
	ToBeAnalyzed []int64
}

/*
# ===========================================================================================
# function convertUnicodeHex
# brief description:
# 	Convert a hex representation string of unicode to the unicode.
# input:
#	s: The hex representation.
# output:
#	The unicode.
*/
func convertUnicodeHex(s []byte) []byte {
	ss := s[3 : len(s)-1]
	u, err := strconv.ParseInt(string(ss), 16, 64)
	if err != nil {
		return []byte("<?>")
	} else {
		return []byte(fmt.Sprintf("%c", u))
	}
}

/*
# ===========================================================================================
# function convertUnicodeDec
# brief description:
# 	Convert a dec representation string of unicode to the unicode.
# input:
#	s: The dec representation.
# output:
#	The unicode.
*/
func convertUnicodeDec(s []byte) []byte {
	ss := s[2 : len(s)-1]
	u, err := strconv.ParseInt(string(ss), 10, 64)
	if err != nil {
		return []byte("<?>")
	} else {
		return []byte(fmt.Sprintf("%c", u))
	}
}

/*
# ===========================================================================================
# function TidyTitle
# brief description: Make a title more tidy before using it. The procedure include the
#                    steps:
#                    (1) remove the spaces at the head and the tail of the title,
#                    (2) replace "&lt;" with "<",
#                    (3) replace "&gt;" with ">",
#                    (4) replace "&amp;" with "&",
#                    (5) replace "&quot;" with "\"",
#                    (6) replace "&apos;" with "'",
#                    (7) replace "&#[number];" with a corresponding unicode
# input:
#   title: The String (text) of a title.
# output:
#   A string of the tidied-up title.
*/
func TidyTitle(title string) string {
	// --------------------------------------------------------------------------------------
	// step 1: remove the spaces at the head and the tail
	result := strings.TrimSpace(title)

	// --------------------------------------------------------------------------------------
	// step 2: replace "&lt;" with "<"
	result = strings.Replace(result, "&lt;", "<", -1)

	// --------------------------------------------------------------------------------------
	// step 3: replace "&gt;" with ">"
	result = strings.Replace(result, "&gt;", ">", -1)

	// --------------------------------------------------------------------------------------
	// step 4: replace "&amp;" with "&"
	result = strings.Replace(result, "&amp;", "&", -1)

	// --------------------------------------------------------------------------------------
	// step 5: replace "&quot;" with "\""
	result = strings.Replace(result, "&quot;", "\"", -1)

	// --------------------------------------------------------------------------------------
	// step 6: replace "&apos;" with "'"
	result = strings.Replace(result, "&apos;", "'", -1)

	// --------------------------------------------------------------------------------------
	// step 7: replace "&#[number];" with a corresponding unicode
	byteResult := []byte(result)
	byteResult = reUnicodeHex.ReplaceAllFunc(byteResult, convertUnicodeHex)
	byteResult = reUnicodeDec.ReplaceAllFunc(byteResult, convertUnicodeDec)
	result = string(byteResult)

	// --------------------------------------------------------------------------------------
	// step 8: return the result
	return result
}

/*
# ===========================================================================================
# function LoadCitationGraph
# brief description:
#	Load data from three files (nodes, edges, labels) of a citation graph.
# input:
#   path: The name of the path where the three files are stored.
#   prefix: The prefix of the names of the three files. For example, the prefix of
#           ijcai-citation-graph-nodes.csv is ijcai.
# output:
#    The citation graph represented by the three files.
*/
func LoadCitationGraph(path string, prefix string) *CitationGraph {
	// --------------------------------------------------------------------------------------
	// step 1: Prepare the data structure of the result.
	result := new(CitationGraph)
	result.Nodes = make(map[int64]*CitationNode)

	// --------------------------------------------------------------------------------------
	// step 2: Assemble the file names for nodes, edges, and labels.
	fileNameOfNodes := path + "/" + prefix + "-citation-graph-nodes.csv"
	fileNameOfEdges := path + "/" + prefix + "-citation-graph-edges.csv"
	fileNameOfLabels := path + "/" + prefix + "-citation-graph-labels.csv"

	// --------------------------------------------------------------------------------------
	// step 3: Load the nodes.
	// (3.1) open the file of Nodes
	fileOfNodes, err := os.Open(fileNameOfNodes)
	if err != nil {
		log.Fatal(err)
	}
	defer fileOfNodes.Close()
	scannerOfNodes := bufio.NewScanner(fileOfNodes)

	// (3.2) Examine the first line to check whether the file format is correct.
	if !scannerOfNodes.Scan() {
		log.Fatal("Cannot read " + fileNameOfNodes)
	}
	firstLine := scannerOfNodes.Text()
	columnNames := strings.Split(firstLine, ",")
	if len(columnNames) != 4 {
		log.Fatal("Incorrect file format of " + fileNameOfNodes)
	}
	if strings.TrimSpace(columnNames[0]) != "#id" ||
		strings.TrimSpace(columnNames[1]) != "in-"+prefix ||
		strings.TrimSpace(columnNames[2]) != "year" ||
		strings.TrimSpace(columnNames[3]) != "title" {
		log.Fatal("Incorrect file format of " + fileNameOfNodes)
	}

	// (3.3) Read the rest of lines.
	for scannerOfNodes.Scan() {
		line := scannerOfNodes.Text()
		columns := strings.Split(line, ",")
		for i := 1; i < 4; i++ {
			columns[i] = strings.TrimSpace(columns[i])
		}
		id, _ := strconv.ParseInt(columns[0], 10, 64)
		isMainNode, _ := strconv.ParseBool(columns[1])
		year, _ := strconv.ParseInt(columns[2], 10, 64)
		title := strings.ReplaceAll(columns[3], "[comma]", ",")
		node := new(CitationNode)
		node.ID = id
		node.Year = year
		node.Title = title
		result.Nodes[id] = node
		if isMainNode {
			result.ToBeAnalyzed = append(result.ToBeAnalyzed, id)
		}
	}

	// --------------------------------------------------------------------------------------
	// step 4: Load the edges.
	// (4.1) Open the file of edges
	fileOfEdges, err := os.Open(fileNameOfEdges)
	if err != nil {
		log.Fatal(err)
	}
	defer fileOfEdges.Close()
	scannerOfEdges := bufio.NewScanner(fileOfEdges)

	// (4.2) Examine the first line to check whether the file format is correct.
	if !scannerOfEdges.Scan() {
		log.Fatal("Cannot read " + fileNameOfEdges)
	}
	firstLine = scannerOfEdges.Text()
	columnNames = strings.Split(firstLine, ",")
	if len(columnNames) != 2 {
		log.Fatal("Incorrect file format of " + fileNameOfEdges)
	}
	if strings.TrimSpace(columnNames[0]) != "#id" ||
		strings.TrimSpace(columnNames[1]) != "ref-id" {
		log.Fatal("Incorrect file format of " + fileNameOfEdges)
	}

	// (4.3) read the rest of lines
	for scannerOfEdges.Scan() {
		line := scannerOfEdges.Text()
		columns := strings.Split(line, ",")
		for i := 1; i < 2; i++ {
			columns[i] = strings.TrimSpace(columns[i])
		}
		id, _ := strconv.ParseInt(columns[0], 10, 64)
		refID, _ := strconv.ParseInt(columns[1], 10, 64)
		node, _ := result.Nodes[id]
		refNode, _ := result.Nodes[refID]
		node.Refs = append(node.Refs, refID)
		refNode.Cites = append(refNode.Cites, id)
	}

	// --------------------------------------------------------------------------------------
	// step 5: Load the labels.
	// (5.1) Open the file of labels
	fileOfLabels, err := os.Open(fileNameOfLabels)
	if err != nil {
		log.Fatal(err)
	}
	defer fileOfLabels.Close()
	scannerOfLabels := bufio.NewScanner(fileOfLabels)

	// (5.2) Examine the first line to check whether the file format is correct.
	if !scannerOfLabels.Scan() {
		log.Fatal("Cannot read " + fileNameOfLabels)
	}
	firstLine = scannerOfLabels.Text()
	columnNames = strings.Split(firstLine, ",")
	if len(columnNames) != 2 {
		log.Fatal("Incorrect file format of " + fileNameOfLabels)
	}
	if strings.TrimSpace(columnNames[0]) != "#id" ||
		strings.TrimSpace(columnNames[1]) != "label" {
		log.Fatal("Incorrect file format of " + fileNameOfEdges)
	}

	// (5.3) Read the rest of lines.
	for scannerOfLabels.Scan() {
		line := scannerOfLabels.Text()
		columns := strings.Split(line, ",")
		for i := 1; i < 2; i++ {
			columns[i] = strings.TrimSpace(columns[i])
		}
		id, _ := strconv.ParseInt(columns[0], 10, 64)
		label := strings.TrimSpace(columns[1])
		node, _ := result.Nodes[id]
		node.Labels = append(node.Labels, label)
	}

	// --------------------------------------------------------------------------------------
	// step 6: Return the result
	return result
}

/*
# ===========================================================================================
# function SaveCitationGraph
# brief description: Save data to three files (nodes, edges, labels) of a citation graph.
# input:
#   path: The name of the path where the three files are stored.
#   prefix: The prefix of the names of the three files. For example, the prefix of
#           ijcai-citation-graph-nodes.csv is ijcai.
#   citationGraph: The citation graph represented by the three files.
# output:
#   nothing
*/
func SaveCitationGraph(path string, prefix string, citationGraph *CitationGraph) {
	// --------------------------------------------------------------------------------------
	// step 1: Assemble the file names for nodes, edges, and labels.
	fileNameOfNodes := path + "/" + prefix + "-citation-graph-nodes.csv"
	fileNameOfEdges := path + "/" + prefix + "-citation-graph-edges.csv"
	fileNameOfLabels := path + "/" + prefix + "-citation-graph-labels.csv"

	// --------------------------------------------------------------------------------------
	// step 2: Save the nodes.
	// (2.1) Open the file of Nodes for writing
	fileOfNodes, err := os.Create(fileNameOfNodes)
	if err != nil {
		log.Fatal(err)
	}
	defer fileOfNodes.Close()

	// (2.2) print the first line with the file format info
	_, err = fileOfNodes.WriteString("#id, in-" + prefix + ", year, title\n")
	if err != nil {
		log.Fatal(err)
	}

	// (2.3) save data into the rest of lines
	idSetForAnalysis := make(map[int64]int)
	for i, v := range citationGraph.ToBeAnalyzed {
		idSetForAnalysis[v] = i
	}
	for id, node := range citationGraph.Nodes {
		_, toBeAnalyzed := idSetForAnalysis[id]
		year := node.Year
		title := TidyTitle(strings.ReplaceAll(node.Title, ",", "[comma]"))
		_, err = fileOfNodes.WriteString(fmt.Sprintf("%d, %t, %d, %s\n", id, toBeAnalyzed, year, title))
		if err != nil {
			log.Fatal(err)
		}
	}

	// --------------------------------------------------------------------------------------
	// step 3: Save the edges.
	// (3.1) Open the file of Edges for writing
	fileOfEdges, err := os.Create(fileNameOfEdges)
	if err != nil {
		log.Fatal(err)
	}
	defer fileOfEdges.Close()

	// (3.2) print the first line with the file format info
	_, err = fileOfEdges.WriteString("#id, ref-id\n")
	if err != nil {
		log.Fatal(err)
	}

	// (3.3) save data into the rest of lines
	edgeSet := make(map[int64]map[int64]bool)
	for id, node := range citationGraph.Nodes {
		setOfID, exists := edgeSet[id]
		if !exists {
			setOfID = make(map[int64]bool)
			edgeSet[id] = setOfID
		}
		for _, refID := range node.Refs {
			setOfID[refID] = true
		}
		for _, citeID := range node.Cites {
			setOfCiteID, exists := edgeSet[citeID]
			if !exists {
				setOfCiteID = make(map[int64]bool)
				edgeSet[citeID] = setOfCiteID
			}
			setOfCiteID[id] = true
		}
	}
	for id, setOfID := range edgeSet {
		for refID, _ := range setOfID {
			fileOfEdges.WriteString(fmt.Sprintf("%d, %d\n", id, refID))
		}
	}

	// --------------------------------------------------------------------------------------
	// step 4: Save the labels.
	// (4.1) Open the file of labels for writing
	fileOfLabels, err := os.Create(fileNameOfLabels)
	if err != nil {
		log.Fatal(err)
	}
	defer fileOfLabels.Close()

	// (4.2) print the first line with the file format info
	_, err = fileOfLabels.WriteString("#id, label\n")
	if err != nil {
		log.Fatal(err)
	}

	// (4.3) save data into the rest of lines
	for id, node := range citationGraph.Nodes {
		for _, label := range node.Labels {
			fileOfLabels.WriteString(fmt.Sprintf("%d, %s\n", id, label))
		}
	}
}
