###### MOSTA 2 SpaCeNet
#
# PURPOSE: Merge Data from the MOSTA Project to SpaCeNet
# AUTHOR: dvo
#
# Convert the segmented cell files containing gene x y umi_count cellId
# to the matching SpaCeNet Format
#
#

# IMPORTS
import pickle
import numpy as np
import sqlite3 as sl
import math as m
import os


# PROCS

# initialize the necessary Database structure
# RETURNS: Database connection
def initDatabase(dbName):
    print('### Initializing Database ###')
    print('-> Connection to local db file')
    con = sl.connect(dbName)

    print('-> Dropping existing tables')
    con.execute("DROP TABLE IF EXISTS t_CELL;")
    con.execute("DROP TABLE IF EXISTS t_CELLPARTS;")
    con.execute("DROP TABLE IF EXISTS t_CELLGENES;")
    con.execute("DROP TABLE IF EXISTS t_GENES;")
    con.execute("DROP TABLE IF EXISTS t_GENEPOS")
    con.execute("DROP TABLE IF EXISTS t_CELLDIST;")

    print('-> Setting up database structure')
    CREATE_TABLE_CELL = "CREATE TABLE t_CELL ( CellId INTEGER, ComX DOUBLE, ComY DOUBLE );"
    CREATE_TABLE_CELLGENES = "CREATE TABLE t_CELLGENES ( CellId INTEGER, Gene TEXT, GeneCnt INTEGER );"
    CREATE_TABLE_CELLPARTS = "CREATE TABLE t_CELLPARTS ( CellId INTEGER, X INTEGER, Y INTEGER );"
    CREATE_TABLE_GENES = "CREATE TABLE t_GENES ( GeneID INTEGER PRIMARY KEY AUTOINCREMENT, GeneName TEXT );"
    CREATE_TABLE_GENEPOS = "CREATE TABLE t_GENEPOS ( GeneName TEXT, MIDCnt INTEGER, X INTEGER, Y INTEGER );"
    CREATE_TABLE_CELLDIST = "CREATE TABLE t_CELLDIST ( CellId_start INTEGER, CellId_end INTEGER, dist DOUBLE ); "

    print('-> Creating tables')
    con.execute(CREATE_TABLE_CELL)
    con.execute(CREATE_TABLE_CELLGENES)
    con.execute(CREATE_TABLE_CELLPARTS)
    con.execute(CREATE_TABLE_GENES)
    con.execute(CREATE_TABLE_GENEPOS)
    con.execute(CREATE_TABLE_CELLDIST)

    print('-> Create math functions')
    con.create_function('SQRT', 1, m.sqrt)
    con.create_function('POW', 2, m.pow)

    print('-> Done')
    return con


# Creates Indizes on specific columns of tables in order to make selects faster
def createDBIndizes(con):
    print("### Create Database Indizes ###")

    # CellId on t_CELLS
    s = "CREATE INDEX cells_index ON t_CELL(CellId)"
    con.execute(s)

    # CellGenes
    s = "CREATE INDEX cellgenes_index ON t_CELLGENES(CellId, Gene)"
    con.execute(s)

    # Genes
    s = "CREATE INDEX genes ON t_GENES(GeneID, GeneName)"
    con.execute(s)


def closeDb(con):
    print("### Close DB Connection ###")
    con.close()


# Reads Coordinates and Genes from DataFile and writes it to t_CELLPARTS, t_GENEPOS and t_GENES
def readCoordinatesAndGenes(path, con):
    print("### Read cell coordinates and Genes ###")

    # Read Spot2Cell.txt
    with open(path, 'r') as stc:
        # Find out number of lines to allocate numpy array
        nrLines = len(stc.readlines())
        print("-> Total number of entries in file: " + str(nrLines - 1))
        # Need to go back to beginning
        stc.seek(0)

        # Read first line (header)
        stc.readline()

        cellCoords = np.empty((nrLines, 3))

        # Counter
        i = 0
        printProgressBar(i, nrLines - 1)

        # REMARK: Adaption to the SCS Output
        # To adapt this again to the output of the SCS paper, one needs to change this routine
        #
        for line in stc:
            gene, strx, stry, UMICnt, strcellId = line.split()
            cellId = int(float(strcellId))
            x = int(strx)
            y = int(stry)
            cellCoords[i, :] = np.array([cellId, x, y])

            # If cell id is 0 then this gene is not associated with any cell, therefore skip it!
            if cellId == 0:
                i += 1
                printProgressBar(i, nrLines - 1)
                continue

            # Write this to db
            INSERT_CELLPARTS = "INSERT into t_CELLPARTS (CellId, X, Y) values ( \'{0}\', {1}, {2});".format(str(cellId),
                                                                                                            str(x),
                                                                                                            str(y))
            con.execute(INSERT_CELLPARTS)

            INSERT_GENEPOS = "INSERT into t_GENEPOS ( GeneName, MIDCnt, X, Y ) "
            INSERT_GENEPOS += " VALUES ( \'" + gene + "\', " + str(UMICnt) + "," + str(x) + ", " + str(y) + ");"

            con.execute(INSERT_GENEPOS)

            printProgressBar(i, nrLines - 1)
            i += 1

        print("")
        # Insert Genes into t_GENES
        print("### Insert Genes into Gene table ###")
        INSERT_GENES = "INSERT into t_GENES ( GeneName ) "
        INSERT_GENES += " SELECT DISTINCT GeneName FROM t_GENEPOS  "
        con.execute(INSERT_GENES)

        return cellCoords


# Calculates the CenterOfMass and fills table t_CELL
def calculateCenterOfMass(cellCoords, con):
    print("### Calculate Center of Mass ###")

    # We need to calculate the "middle of the cell"
    # therefor: x_mid = 1/nx * sum over all x

    # we need to get all x and y coordinates for one cell
    # therefor first get the cell count

    cellCnt = int(float(np.amax(cellCoords, axis=0)[0]))
    # print('Number of Cells: ' + str(cellCnt[0]))

    # Instantiate a array for all cells with their center of mass
    comCells = np.empty((cellCnt, 3))

    # In the next step, we need to run through all cells and sum over all x and y values
    i = 0
    printProgressBar(i, cellCnt)

    for cell in range(cellCnt):
        rows = np.where(cellCoords[:, 0] == cell + 1)
        cnt = len(rows[0])

        if cnt != 0:
            tmp, x, y = np.sum(cellCoords[rows], axis=0) / cnt

            INSERT_CELL = "INSERT into t_CELL( CellId, ComX, ComY ) values ( \'{0}\', {1}, {2});".format(str(cell + 1),
                                                                                                         str(x), str(y))

            con.execute(INSERT_CELL)

        printProgressBar(i, cellCnt)
        i += 1
    print("")


# Merge Cells and corresponding genes into t_CELLGENES
def getGenesForCells(con):
    print("### Merging Cells and Genes ###")

    INSERT_CELLGENES = "INSERT into t_CELLGENES ( CellId, Gene, GeneCnt) "
    INSERT_CELLGENES += " SELECT cp.CellId, gp.GeneName, sum(gp.MIDCnt)"
    INSERT_CELLGENES += " FROM t_CELLPARTS cp, t_GENEPOS gp "
    INSERT_CELLGENES += " WHERE cp.X = gp.X "
    INSERT_CELLGENES += " AND cp.Y = gp.Y "
    INSERT_CELLGENES += " GROUP BY cp.CellId, gp.GeneName; "

    con.execute(INSERT_CELLGENES)


# Calculate Distances between different Cells and store them in t_CELLDIST
def calculateCellDistance(con):
    print("### Calculate Distance between Cells ###")

    # Select all Cell Ids and their Center of mass, and run through all rows
    SELECT_ALL_CELLS = "SELECT CellId, ComX, ComY FROM t_CELL; "
    data = con.execute(SELECT_ALL_CELLS)

    # Now calculate the distance with each other cell using euclidean distance
    # dist(A,B) = sqrt( (a_x + b_x)^2 + (a_y+b_y)^2)
    for row in data:
        INSERT_DISTANCE = "INSERT into t_CELLDIST( CellId_start, CellId_end, dist ) "
        INSERT_DISTANCE += "SELECT " + str(row[0]) + ", "
        INSERT_DISTANCE += "c.CellId, "
        INSERT_DISTANCE += "SQRT( POW( " + str(row[1]) + " - c.ComX, 2) + POW( " + str(row[2]) + " - c.ComY , 2)) "
        INSERT_DISTANCE += "FROM t_CELL c"

        con.execute(INSERT_DISTANCE)


# Write the information from the database to a pickle file
def writeOutput(output, nrCells, geneEnd, geneStart, geneIdList, con, geneOutput="data/Genes.txt",
                fStartFromCellWithMostGenes=False):
    print("### Writing Output ###")
    print("### Exporting " + str(nrCells) + " cells with " + str(geneEnd - geneStart) + " genes ###")
    coord_mat = np.zeros(shape=(1, nrCells, 3), dtype=float)
    coord_mat[:, :, 2] = 0  # set z direction to one as we only have a two dim tissue here

    X_mat = np.zeros(shape=(1, nrCells, geneEnd - geneStart), dtype=float)

    # Store Gene Names
    GeneNames = np.empty((geneEnd - geneStart, 2), dtype=object)
    # genes = con.execute("select GeneID, GeneName from t_GENES where GeneID between " + str(geneStart) + " and " + str(geneEnd))

    genes = con.execute("select GeneID, GeneName from t_GENES where GeneID in " + geneIdList + " order by 1 asc")

    with open(geneOutput, 'w') as f:
        counter = 0
        f.write("###\t\t Export Info and Genes \t\t###\n")
        f.write("-\t\t Nr of Cells: " + str(nrCells) + "\t\t-\n")
        f.write("-\t\t Exported Genes (geneId:GeneName) \t\t-\n")
        for gene in genes:
            GeneNames[counter, 1] = gene[1]
            GeneNames[counter, 0] = gene[0]
            counter += 1
            f.write(str(gene[0]) + ":" + str(gene[1]) + "\n")
            # print(str(gene[0]) + ":" + str(gene[1]))
    print("-> Saved all Genes in " + str(geneOutput))

    bestSpotCellId = 0
    # Find a spot, where most of these genes exists and get its CellNr
    if fStartFromCellWithMostGenes:
        s = "SELECT cg.CellID, SUM(cg.GeneCnt) FROM t_CELLGENES cg, t_GENES g WHERE cg.Gene = g.GeneName and g.GeneID in " + geneIdList + " GROUP BY CellID ORDER BY 2 desc"
        data = con.execute(s)
        bestSpotCellId, geneCnt = data.fetchone()
        print("-> Most matching Cell: " + str(bestSpotCellId) + " with Nr of Matching Genes " + str(geneCnt))

    counter = 0
    cells = con.execute("select CellId, ComX, ComY from t_CELL WHERE CellId >= "
                        "" + str(bestSpotCellId) + " ORDER BY CellId asc LIMIT " + str(nrCells))

    i = 0
    printProgressBar(i, nrCells)

    for cell in cells:
        coord_mat[0, counter, 0] = cell[1]  # X
        coord_mat[0, counter, 1] = cell[2]  # Y

        gcounter = 0
        for gene in GeneNames:
            # Get genes for this cell
            s = "select g.GeneID, cg.GeneCnt, g.GeneName from t_CELLGENES cg LEFT JOIN t_GENES g on cg.Gene = g.GeneName "
            s += "WHERE cg.CellId = " + str(cell[0])
            s += " AND g.GeneId = " + str(gene[0])
            data = con.execute(s)
            gene = data.fetchone()

            if gene is None:
                X_mat[0, counter, gcounter] = 0
            else:
                X_mat[0, counter, gcounter] = gene[1]
            gcounter += 1

        counter += 1

        printProgressBar(i, nrCells)
        i += 1

    print("")
    print("### Dump data ###")
    np.savez(output, X_mat=X_mat, coord_mat=coord_mat, GeneNames=GeneNames)

    """# Write this to a pickle file for further use
    print("### Dump data ###")
    data = {"X_mat": X_mat, "coord_mat": coord_mat, "GeneNames": GeneNames}
    with open(output, 'wb') as f:
        pickle.dump(data, f)"""


# Get total number of cells, Returns Integer
def getCellNumber(con):
    print("### Get number of cells ###")

    s = "SELECT COUNT(*) from t_CELL"
    data = con.execute(s)
    nrOfCells = data.fetchone()[0]

    print("-> Total Number of Cells: " + str(nrOfCells))

    return int(nrOfCells)


def getGeneIdForName(Name, con):
    print("### Get ID for Gene " + str(Name) + " ###")
    s = "SELECT g.GeneID from t_GENES g where g.GeneName like \'" + str(Name) + "\'"
    data = con.execute(s)

    return data.fetchone()[0]


# Print iterations progress
def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


def export(path='',
           output='',
           GeneList=[],
           cellPercentage=0.20,
           dbName="mosta-to-spacenet.db",
           outputExportInfo="data/Genes.txt"):
    print('### Start SCS2SpaCeNet ###')

    if not os.path.exists(dbName):
        print("-> Database will be created")
        con = initDatabase(dbName)

        cellCoords = readCoordinatesAndGenes(path, con)
        calculateCenterOfMass(cellCoords, con)
        getGenesForCells(con)
        createDBIndizes(con)

        con.commit()
    else:
        print("-> Export Mode was called")
        print("-> Existing Database will be kept")
        con = sl.connect(dbName)

    # Exporting data
    geneStart = 0

    # Get total number of cells
    totalNumberCells = getCellNumber(con)
    lowerCellLimit = np.round(cellPercentage * totalNumberCells, decimals=0)
    SELECT_GENES = "SELECT cg.Gene, count(*) "
    SELECT_GENES += "FROM t_CELLGENES cg GROUP BY cg.Gene "
    SELECT_GENES += "HAVING count(*) >= "
    SELECT_GENES += str(lowerCellLimit)
    SELECT_GENES += ""
    print(SELECT_GENES)
    data = con.execute(SELECT_GENES)

    for row in data:
        GeneList.append(str(row[0]))

    nrOfGenesToExport = len(GeneList)

    print("Genes to export: " + str(nrOfGenesToExport))

    geneIdList = ""
    for i in range(nrOfGenesToExport):
        geneIdList += str(getGeneIdForName(GeneList[i], con)) + ", "

    geneIdList = "(" + geneIdList[:len(geneIdList) - 2] + ")"
    geneEnd = nrOfGenesToExport

    nrCells = getCellNumber(con)
    writeOutput(output+".npz", nrCells, geneEnd, geneStart, geneIdList, con, geneOutput=output+".txt")

    closeDb(con)




if __name__ == '__main__':
    """
    docker run -it --rm -v /sybig/home/ssc/SpaCeNet_GitHub:/mnt spacenet python3 preprocess_MOSTA.py
    """
    cellPercentage = 0.30
    export(path="data/SS200000135TL_D1_CellBin.tsv",
           output=f"data/MouseBrainAdult_{int(cellPercentage*100)}Percent",
           cellPercentage=cellPercentage,
           dbName="data/MouseBrainAdult_complete.db"
           )
