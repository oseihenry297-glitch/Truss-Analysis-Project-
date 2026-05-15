# -*- coding: utf-8 -*-
"""
Created on Mon May 11 22:16:08 2026

@author: henos
"""

# ================================================
# 2D Truss Analysis App - Method of Joints
# KNUST Civil Engineering - Year 2
# ================================================

import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


# ================================================
# CLASSES
# ================================================

class Joint:
    """Stores the properties of a single joint in the truss"""

    def __init__(self, name, x, y):
        self.name    = name       # joint label e.g. 'A'
        self.x       = float(x)  # x-coordinate in metres
        self.y       = float(y)  # y-coordinate in metres
        self.fx      = 0.0       # applied load in x-direction (N)
        self.fy      = 0.0       # applied load in y-direction (N)
        self.support = "free"    # "free", "pin", or "roller"


class Member:
    """Stores the properties of a bar connecting two joints"""

    def __init__(self, startJoint, endJoint):
        self.startJoint = startJoint
        self.endJoint   = endJoint
        self.name       = startJoint.name + "-" + endJoint.name
        self.A          = 0.01    # cross-section area in m^2 (steel)
        self.E          = 200e9   # Young's modulus in Pa (steel)

    def getLength(self):
        """Returns the length of the member using Pythagoras"""
        dx = self.endJoint.x - self.startJoint.x
        dy = self.endJoint.y - self.startJoint.y
        length = np.sqrt(dx**2 + dy**2)
        return length

    def getCosines(self, fromJoint):
        """
        Returns (cx, cy) — direction cosines pointing
        away from fromJoint toward the other joint.
        These are used in the equilibrium equations.
        """
        if fromJoint.name == self.startJoint.name:
            dx = self.endJoint.x - self.startJoint.x
            dy = self.endJoint.y - self.startJoint.y
        else:
            dx = self.startJoint.x - self.endJoint.x
            dy = self.startJoint.y - self.endJoint.y

        L  = self.getLength()
        cx = dx / L
        cy = dy / L
        return cx, cy

# ================================================
# GLOBAL LISTS
# ================================================

jointList  = []   # list of Joint objects
memberList = []   # list of Member objects


# ================================================
# HELPER FUNCTION
# ================================================

def findJoint(name):
    """Searches jointList and returns the joint with the given name"""
    for j in jointList:
        if j.name == name:
            return j
    return None   # return None if not found


# ================================================
# SOLVER - Method of Joints (Matrix Form)
# ================================================

def solveTruss(virtualLoad=None):
    """
    Builds and solves the equilibrium equations for the truss.

    For every joint we write two equations:
        Sum of Fx = 0
        Sum of Fy = 0

    Unknowns = member forces + reaction forces at supports

    virtualLoad: dict {jointName: [Fx, Fy]}
                 Used only when computing deflections.
                 If None, the real applied loads are used.

    Returns the solution array or None if system cannot be solved.
    """

    n = len(jointList)   # number of joints
    m = len(memberList)  # number of members

    # Build list of reaction unknowns based on support types
    reactionList = []   # each entry is (joint, "x" or "y")
    for j in jointList:
        if j.support == "pin":
            reactionList.append((j, "x"))
            reactionList.append((j, "y"))
        elif j.support == "roller":
            reactionList.append((j, "y"))

    # Total unknowns = member forces + reactions
    totalUnknowns = m + len(reactionList)

    # Create the matrix A and vector b (A * unknowns = b)
    A = np.zeros((2*n, totalUnknowns))
    b = np.zeros(2*n)

    for ji, joint in enumerate(jointList):

        rowX = 2 * ji       # row index for Sum Fx = 0
        rowY = 2 * ji + 1   # row index for Sum Fy = 0

        # Add member force contributions
        for mi, mem in enumerate(memberList):
            if joint.name in [mem.startJoint.name, mem.endJoint.name]:
                cx, cy = mem.getCosines(joint)
                A[rowX, mi] = cx
                A[rowY, mi] = cy

        # Add reaction force contributions
        for ri, (rJoint, dof) in enumerate(reactionList):
            if rJoint.name == joint.name:
                if dof == "x":
                    A[rowX, m + ri] = 1.0
                else:
                    A[rowY, m + ri] = 1.0

        # Set the load values on the right hand side
        if virtualLoad and joint.name in virtualLoad:
            loadFx = virtualLoad[joint.name][0]
            loadFy = virtualLoad[joint.name][1]
        else:
            loadFx = joint.fx
            loadFy = joint.fy

        b[rowX] = -loadFx
        b[rowY] = -loadFy

    # Check if system can be solved (determinant must not be zero)
    if abs(np.linalg.det(A)) < 1e-10:
        messagebox.showerror("Solver Error",
            "System cannot be solved.\n"
            "Check your supports and members.")
        return None

    solution = np.linalg.solve(A, b)
    return solution


# ================================================
# DEFLECTION - Virtual Work Method
# ================================================

def getDeflection(targetJointName, direction):
    """
    Calculates deflection at a joint using the Virtual Work Method.

    Formula:  delta = Sum( F * f * L / (A * E) )

    Where:
        F = real member force from applied loads
        f = virtual member force from a unit load at target joint
        L = member length
        A = cross-section area
        E = Young's modulus
    """

    # Step 1: Solve for real member forces
    realSolution = solveTruss()
    if realSolution is None:
        return None

    m = len(memberList)
    realForces = realSolution[:m]   # first m values are member forces

    # Step 2: Apply a virtual unit load at the target joint
    if direction == "x":
        virtualLoad = {targetJointName: [1.0, 0.0]}
    else:
        virtualLoad = {targetJointName: [0.0, 1.0]}

    virtualSolution = solveTruss(virtualLoad)
    if virtualSolution is None:
        return None

    virtualForces = virtualSolution[:m]

    # Step 3: Apply the virtual work formula
    delta = 0.0
    for i in range(m):
        F = realForces[i]
        f = virtualForces[i]
        L = memberList[i].getLength()
        A = memberList[i].A
        E = memberList[i].E
        delta += (F * f * L) / (A * E)

    return delta


# ================================================
# PLOT FUNCTION
# ================================================

def drawTruss(memberForces=None, deflections=None):     
    """
    Draws the truss on the matplotlib canvas.
    - memberForces: list of axial forces (blue=tension, red=compression)
    - deflections: dict {jointName: (dx, dy)} for deformed shape
    """

    # Clear old plot
    for widget in plotFrame.winfo_children():
        widget.destroy()

    fig = Figure(figsize=(8, 6), dpi=100)
    ax  = fig.add_subplot(111)

    # Draw each member (original position)
    for i, mem in enumerate(memberList):
        x1 = mem.startJoint.x
        y1 = mem.startJoint.y
        x2 = mem.endJoint.x
        y2 = mem.endJoint.y

        if memberForces is not None:
            F     = memberForces[i]
            color = "blue" if F >= 0 else "red"
            state = "T" if F >= 0 else "C"
            xMid  = (x1 + x2) / 2
            yMid  = (y1 + y2) / 2
            ax.text(xMid, yMid + 0.05, f"{F:.1f}N ({state})",
                    fontsize=8, color=color, ha="center")
        else:
            color = "black"

        ax.plot([x1, x2], [y1, y2], color=color, linewidth=2)

    # Draw deformed shape (dashed lines)
    if deflections is not None:
        scale = 1000   # amplify deflections so they are visible

        deformedPos = {}
        for j in jointList:
            dx, dy = deflections[j.name]
            deformedPos[j.name] = (j.x + dx * scale,
                                   j.y + dy * scale)

        for mem in memberList:
            x1d, y1d = deformedPos[mem.startJoint.name]
            x2d, y2d = deformedPos[mem.endJoint.name]
            ax.plot([x1d, x2d], [y1d, y2d],
                    "k--", linewidth=1.5, alpha=0.6)

        ax.plot([], [], "k--", linewidth=1.5,
                label="Deformed shape (x" + str(scale) + ")")

    # Draw joints as dots with labels
    for j in jointList:
        ax.scatter(j.x, j.y, color="black", s=50, zorder=5)
        ax.text(j.x + 0.04, j.y + 0.07, j.name,
                fontsize=10, fontweight="bold")

    # Draw support symbols
    for j in jointList:
        if j.support == "pin":
            ax.scatter(j.x, j.y - 0.2, marker="^",
                       s=200, color="purple")
            ax.text(j.x, j.y - 0.38, "PIN",
                    fontsize=7, color="purple", ha="center")
        elif j.support == "roller":
            ax.scatter(j.x, j.y - 0.2, marker="o",
                       s=100, color="orange")
            ax.text(j.x, j.y - 0.38, "ROLLER",
                    fontsize=7, color="orange", ha="center")

    # Draw load arrows
    for j in jointList:
        if j.fx != 0 or j.fy != 0:
            mag   = np.sqrt(j.fx**2 + j.fy**2)
            scale2 = 0.3 / mag
            ax.annotate("",
                xy=(j.x + j.fx * scale2,
                    j.y + j.fy * scale2),
                xytext=(j.x, j.y),
                arrowprops=dict(arrowstyle="-|>",
                                color="green", lw=2))

    # Legend
    if memberForces is not None:
        ax.plot([], [], color="blue", lw=2, label="Tension (+)")
        ax.plot([], [], color="red",  lw=2, label="Compression (-)")

    ax.legend(fontsize=8)
    ax.set_title("2D Truss Analysis - Method of Joints",
                 fontsize=13, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.axis("equal")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")

    canvas = FigureCanvasTkAgg(fig, master=plotFrame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)


# ================================================
# BUTTON COMMANDS
# ================================================

def cmdAddJoint():
    """Reads joint inputs and adds a new Joint to jointList"""
    name = entryJointName.get().strip()
    x    = entryJointX.get().strip()
    y    = entryJointY.get().strip()

    if name == "" or x == "" or y == "":
        messagebox.showerror("Error", "Please fill in all joint fields.")
        return

    if findJoint(name):
        messagebox.showerror("Error", "A joint with that name already exists.")
        return

    newJoint = Joint(name, x, y)
    jointList.append(newJoint)

    # Update the deflection dropdown with new joint names
    comboDeflJoint["values"] = [j.name for j in jointList]

    messagebox.showinfo("Success", "Joint " + name + " added.")
    entryJointName.delete(0, tk.END)
    entryJointX.delete(0, tk.END)
    entryJointY.delete(0, tk.END)


def cmdAddMember():
    """Reads member inputs and adds a new Member to memberList"""
    startName = entryMemStart.get().strip()
    endName   = entryMemEnd.get().strip()

    startJoint = findJoint(startName)
    endJoint   = findJoint(endName)

    if startJoint is None or endJoint is None:
        messagebox.showerror("Error", "Joint not found. Add joints first.")
        return

    if startName == endName:
        messagebox.showerror("Error", "Start and End joints must be different.")
        return

    newMember = Member(startJoint, endJoint)
    memberList.append(newMember)

    messagebox.showinfo("Success", "Member " + newMember.name + " added.")
    entryMemStart.delete(0, tk.END)
    entryMemEnd.delete(0, tk.END)


def cmdAddLoad():
    """Reads load inputs and applies them to the specified joint"""
    name = entryLoadJoint.get().strip()
    j    = findJoint(name)

    if j is None:
        messagebox.showerror("Error", "Joint not found.")
        return

    j.fx = float(entryLoadFx.get())
    j.fy = float(entryLoadFy.get())

    messagebox.showinfo("Success",
        "Load applied at joint " + name + ".")


def cmdAddSupport():
    """Reads support inputs and sets the support type on the joint"""
    name = entrySupportJoint.get().strip()
    j    = findJoint(name)

    if j is None:
        messagebox.showerror("Error", "Joint not found.")
        return

    supportType = comboSupportType.get()
    if supportType not in ["pin", "roller"]:
        messagebox.showerror("Error", "Choose pin or roller.")
        return

    j.support = supportType
    messagebox.showinfo("Success",
        supportType + " support added at joint " + name + ".")


def cmdSolveTruss():
    """Solves the truss and displays member forces and reactions"""
    if len(jointList) == 0 or len(memberList) == 0:
        messagebox.showerror("Error", "Please add joints and members first.")
        return

    solution = solveTruss()
    if solution is None:
        return

    m = len(memberList)
    memberForces = solution[:m]   # first m values are member forces

    # Build results text
    resultText = "--- MEMBER FORCES ---\n"
    for i, mem in enumerate(memberList):
        F    = memberForces[i]
        fType = "Tension" if F >= 0 else "Compression"
        resultText += "  " + mem.name + ": " + str(round(F, 2)) + " N (" + fType + ")\n"

    resultText += "\n--- REACTIONS ---\n"
    reactionList = []
    for j in jointList:
        if j.support == "pin":
            reactionList.append((j, "x"))
            reactionList.append((j, "y"))
        elif j.support == "roller":
            reactionList.append((j, "y"))

    for ri, (j, dof) in enumerate(reactionList):
        resultText += "  R_" + j.name + dof + ": " + str(round(solution[m + ri], 2)) + " N\n"

    messagebox.showinfo("Truss Results", resultText)
    drawTruss(memberForces)


def cmdCalcDeflection():
    """Calculates deflection at selected joint and draws deformed shape"""
    targetName = comboDeflJoint.get()
    direction  = comboDeflDir.get()

    if targetName == "" or direction == "":
        messagebox.showerror("Error", "Please select a joint and direction.")
        return

    delta = getDeflection(targetName, direction)
    if delta is None:
        return

    messagebox.showinfo("Deflection Result",
        "Deflection at joint " + targetName + " (" + direction + "-direction):\n\n"
        "  delta = " + "{:.6e}".format(delta) + " m")

    # Compute deflections at all joints for the deformed shape
    realSolution = solveTruss()
    if realSolution is None:
        return

    deflections = {}
    for j in jointList:
        dx = getDeflection(j.name, "x")
        dy = getDeflection(j.name, "y")
        if dx is None: dx = 0.0
        if dy is None: dy = 0.0
        deflections[j.name] = (dx, dy)

    drawTruss(realSolution[:len(memberList)], deflections)


def cmdClearAll():
    """Clears all joints, members and the plot"""
    jointList.clear()
    memberList.clear()
    comboDeflJoint["values"] = []

    for widget in plotFrame.winfo_children():
        widget.destroy()

    messagebox.showinfo("Cleared", "All data has been cleared.")


# ================================================
# GUI LAYOUT
# ================================================

app = tk.Tk()
app.title("2D Truss Analysis App - Method of Joints")
app.geometry("1280x720")

mainFrame = tk.Frame(app)
mainFrame.pack(fill="both", expand=True)

# ---- Left panel (input controls) ----
leftPanel = tk.Frame(mainFrame, padx=10, pady=10, relief="ridge", bd=2)
leftPanel.pack(side="left", fill="y", padx=8, pady=8)

# -- Joints LabelFrame --
jointFrame = tk.LabelFrame(leftPanel, text="Joints",
                            font=("Arial", 10, "bold"), padx=8, pady=6)
jointFrame.pack(fill="x", pady=4)

tk.Label(jointFrame, text="Name").grid(row=0, column=0, sticky="w")
entryJointName = tk.Entry(jointFrame, width=8)
entryJointName.grid(row=0, column=1, padx=4)

tk.Label(jointFrame, text="X (m)").grid(row=1, column=0, sticky="w")
entryJointX = tk.Entry(jointFrame, width=8)
entryJointX.grid(row=1, column=1, padx=4)

tk.Label(jointFrame, text="Y (m)").grid(row=2, column=0, sticky="w")
entryJointY = tk.Entry(jointFrame, width=8)
entryJointY.grid(row=2, column=1, padx=4)

tk.Button(jointFrame, text="Add Joint", command=cmdAddJoint,
          bg="#388e3c", fg="white", width=14
          ).grid(row=3, column=0, columnspan=2, pady=4)

# -- Members LabelFrame --
memberFrame = tk.LabelFrame(leftPanel, text="Members",
                             font=("Arial", 10, "bold"), padx=8, pady=6)
memberFrame.pack(fill="x", pady=4)

tk.Label(memberFrame, text="Start").grid(row=0, column=0, sticky="w")
entryMemStart = tk.Entry(memberFrame, width=8)
entryMemStart.grid(row=0, column=1, padx=4)

tk.Label(memberFrame, text="End").grid(row=1, column=0, sticky="w")
entryMemEnd = tk.Entry(memberFrame, width=8)
entryMemEnd.grid(row=1, column=1, padx=4)

tk.Button(memberFrame, text="Add Member", command=cmdAddMember,
          bg="#388e3c", fg="white", width=14
          ).grid(row=2, column=0, columnspan=2, pady=4)

# -- Loads LabelFrame --
loadFrame = tk.LabelFrame(leftPanel, text="Loads",
                           font=("Arial", 10, "bold"), padx=8, pady=6)
loadFrame.pack(fill="x", pady=4)

tk.Label(loadFrame, text="Joint").grid(row=0, column=0, sticky="w")
entryLoadJoint = tk.Entry(loadFrame, width=8)
entryLoadJoint.grid(row=0, column=1, padx=4)

tk.Label(loadFrame, text="Fx (N)").grid(row=1, column=0, sticky="w")
entryLoadFx = tk.Entry(loadFrame, width=8)
entryLoadFx.grid(row=1, column=1, padx=4)

tk.Label(loadFrame, text="Fy (N)").grid(row=2, column=0, sticky="w")
entryLoadFy = tk.Entry(loadFrame, width=8)
entryLoadFy.grid(row=2, column=1, padx=4)

tk.Button(loadFrame, text="Add Load", command=cmdAddLoad,
          bg="#1565c0", fg="white", width=14
          ).grid(row=3, column=0, columnspan=2, pady=4)

# -- Supports LabelFrame --
supportFrame = tk.LabelFrame(leftPanel, text="Supports",
                              font=("Arial", 10, "bold"), padx=8, pady=6)
supportFrame.pack(fill="x", pady=4)

tk.Label(supportFrame, text="Joint").grid(row=0, column=0, sticky="w")
entrySupportJoint = tk.Entry(supportFrame, width=8)
entrySupportJoint.grid(row=0, column=1, padx=4)

tk.Label(supportFrame, text="Type").grid(row=1, column=0, sticky="w")
comboSupportType = ttk.Combobox(supportFrame, values=["pin", "roller"], width=6)
comboSupportType.grid(row=1, column=1, padx=4)

tk.Button(supportFrame, text="Add Support", command=cmdAddSupport,
          bg="#1565c0", fg="white", width=14
          ).grid(row=2, column=0, columnspan=2, pady=4)

# -- Deflection LabelFrame --
deflFrame = tk.LabelFrame(leftPanel, text="Deflection",
                           font=("Arial", 10, "bold"), padx=8, pady=6)
deflFrame.pack(fill="x", pady=4)

tk.Label(deflFrame, text="Joint").grid(row=0, column=0, sticky="w")
comboDeflJoint = ttk.Combobox(deflFrame, width=6)
comboDeflJoint.grid(row=0, column=1, padx=4)

tk.Label(deflFrame, text="Direction").grid(row=1, column=0, sticky="w")
comboDeflDir = ttk.Combobox(deflFrame, values=["x", "y"], width=6)
comboDeflDir.grid(row=1, column=1, padx=4)

# -- Action Buttons --
tk.Button(leftPanel, text="SOLVE TRUSS", command=cmdSolveTruss,
          bg="#2e7d32", fg="white",
          font=("Arial", 10, "bold"), width=22).pack(pady=4)

tk.Button(leftPanel, text="CALCULATE DEFLECTION", command=cmdCalcDeflection,
          bg="#1565c0", fg="white",
          font=("Arial", 10, "bold"), width=22).pack(pady=4)

tk.Button(leftPanel, text="CLEAR ALL", command=cmdClearAll,
          bg="#c62828", fg="white",
          font=("Arial", 10, "bold"), width=22).pack(pady=4)

# ---- Right panel (plot area) ----
rightPanel = tk.Frame(mainFrame)
rightPanel.pack(side="right", fill="both", expand=True)

plotFrame = tk.LabelFrame(rightPanel, text="Truss Diagram",
                           font=("Arial", 11, "bold"),
                           padx=5, pady=5)
plotFrame.pack(fill="both", expand=True, padx=8, pady=8)

# ================================================
# START THE APP
# ================================================

app.mainloop()