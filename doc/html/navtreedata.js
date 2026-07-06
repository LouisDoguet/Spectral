/*
 @licstart  The following is the entire license notice for the JavaScript code in this file.

 The MIT License (MIT)

 Copyright (C) 1997-2020 by Dimitri van Heesch

 Permission is hereby granted, free of charge, to any person obtaining a copy of this software
 and associated documentation files (the "Software"), to deal in the Software without restriction,
 including without limitation the rights to use, copy, modify, merge, publish, distribute,
 sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in all copies or
 substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
 BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
 DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

 @licend  The above is the entire license notice for the JavaScript code in this file
*/
var NAVTREE =
[
  [ "Spectral1D", "index.html", [
    [ "Implementation Guide: Residual-Based Optimization for Burgers Equation", "da/d28/md_optimesh_2IMPLEMENTATION__GUIDE.html", [
      [ "Overview", "da/d28/md_optimesh_2IMPLEMENTATION__GUIDE.html#autotoc_md1", null ],
      [ "Part 1: Create <tt>burgers.py</tt> (New File)", "da/d28/md_optimesh_2IMPLEMENTATION__GUIDE.html#autotoc_md3", [
        [ "File Location", "da/d28/md_optimesh_2IMPLEMENTATION__GUIDE.html#autotoc_md4", null ],
        [ "Class Structure", "da/d28/md_optimesh_2IMPLEMENTATION__GUIDE.html#autotoc_md5", null ],
        [ "Key Methods to Implement", "da/d28/md_optimesh_2IMPLEMENTATION__GUIDE.html#autotoc_md6", [
          [ "1. <tt>__init__(self, P=10, shock_intensity=10.0)</tt>", "da/d28/md_optimesh_2IMPLEMENTATION__GUIDE.html#autotoc_md7", null ],
          [ "2. <tt>exact_solution(self, x)</tt>", "da/d28/md_optimesh_2IMPLEMENTATION__GUIDE.html#autotoc_md9", null ],
          [ "3. <tt>flux(self, u)</tt>", "da/d28/md_optimesh_2IMPLEMENTATION__GUIDE.html#autotoc_md11", null ],
          [ "4. <tt>compute_rbf_derivative_matrix(self, X, eps_array)</tt>", "da/d28/md_optimesh_2IMPLEMENTATION__GUIDE.html#autotoc_md13", null ],
          [ "5. <tt>compute_rbf_mass_matrix(self, X, eps_array, quad_order=None)</tt>", "da/d28/md_optimesh_2IMPLEMENTATION__GUIDE.html#autotoc_md15", null ],
          [ "6. <tt>compute_residual(self, u_values, X, eps_array)</tt>", "da/d28/md_optimesh_2IMPLEMENTATION__GUIDE.html#autotoc_md17", null ],
          [ "7. <tt>residual_norm_squared(self, u_values, X, eps_array)</tt>", "da/d28/md_optimesh_2IMPLEMENTATION__GUIDE.html#autotoc_md19", null ]
        ] ],
        [ "Summary: What <tt>burgers.py</tt> Contains", "da/d28/md_optimesh_2IMPLEMENTATION__GUIDE.html#autotoc_md21", null ]
      ] ],
      [ "Part 2: Modify <tt>solution.py</tt>", "da/d28/md_optimesh_2IMPLEMENTATION__GUIDE.html#autotoc_md23", [
        [ "What to Add/Change", "da/d28/md_optimesh_2IMPLEMENTATION__GUIDE.html#autotoc_md24", null ]
      ] ],
      [ "Part 3: Modify <tt>optimize.py</tt>", "da/d28/md_optimesh_2IMPLEMENTATION__GUIDE.html#autotoc_md26", [
        [ "Changes Required", "da/d28/md_optimesh_2IMPLEMENTATION__GUIDE.html#autotoc_md27", [
          [ "1. Import Burgers", "da/d28/md_optimesh_2IMPLEMENTATION__GUIDE.html#autotoc_md28", null ],
          [ "2. Create a Burgers instance at module level", "da/d28/md_optimesh_2IMPLEMENTATION__GUIDE.html#autotoc_md29", null ],
          [ "3. Replace the objective function", "da/d28/md_optimesh_2IMPLEMENTATION__GUIDE.html#autotoc_md30", null ],
          [ "4. Update optimize_case() signature", "da/d28/md_optimesh_2IMPLEMENTATION__GUIDE.html#autotoc_md31", null ],
          [ "5. Update run() function", "da/d28/md_optimesh_2IMPLEMENTATION__GUIDE.html#autotoc_md32", null ]
        ] ]
      ] ],
      [ "Part 4: Create <tt>validation.py</tt> (New File)", "da/d28/md_optimesh_2IMPLEMENTATION__GUIDE.html#autotoc_md34", [
        [ "File Location", "da/d28/md_optimesh_2IMPLEMENTATION__GUIDE.html#autotoc_md35", null ],
        [ "What It Does", "da/d28/md_optimesh_2IMPLEMENTATION__GUIDE.html#autotoc_md36", null ]
      ] ],
      [ "Part 5: Integration Checklist", "da/d28/md_optimesh_2IMPLEMENTATION__GUIDE.html#autotoc_md38", [
        [ "Code Structure", "da/d28/md_optimesh_2IMPLEMENTATION__GUIDE.html#autotoc_md39", null ],
        [ "Correctness Checks", "da/d28/md_optimesh_2IMPLEMENTATION__GUIDE.html#autotoc_md40", null ],
        [ "Conditioning Checks", "da/d28/md_optimesh_2IMPLEMENTATION__GUIDE.html#autotoc_md41", null ],
        [ "Convergence Checks", "da/d28/md_optimesh_2IMPLEMENTATION__GUIDE.html#autotoc_md42", null ],
        [ "Plot Quality Checks", "da/d28/md_optimesh_2IMPLEMENTATION__GUIDE.html#autotoc_md43", null ]
      ] ],
      [ "Key Design Decisions Made", "da/d28/md_optimesh_2IMPLEMENTATION__GUIDE.html#autotoc_md45", null ],
      [ "Expected Results", "da/d28/md_optimesh_2IMPLEMENTATION__GUIDE.html#autotoc_md47", null ],
      [ "Notes for C++ Port", "da/d28/md_optimesh_2IMPLEMENTATION__GUIDE.html#autotoc_md49", null ]
    ] ],
    [ "Claude Code: Atomic Implementation Tasks", "d7/d6b/md_optimesh_2TASK__LIST.html", [
      [ "PHASE 1: CREATE BURGERS CLASS (1.5 hours)", "d7/d6b/md_optimesh_2TASK__LIST.html#autotoc_md52", [
        [ "Task 1.1: Create file <tt>optimesh/burgers.py</tt>", "d7/d6b/md_optimesh_2TASK__LIST.html#autotoc_md53", null ],
        [ "Task 1.2: Implement <tt>Burgers.__init__()</tt>", "d7/d6b/md_optimesh_2TASK__LIST.html#autotoc_md54", null ],
        [ "Task 1.3: Implement <tt>Burgers.exact_solution()</tt>", "d7/d6b/md_optimesh_2TASK__LIST.html#autotoc_md55", null ],
        [ "Task 1.4: Implement <tt>Burgers.flux()</tt>", "d7/d6b/md_optimesh_2TASK__LIST.html#autotoc_md56", null ],
        [ "Task 1.5: Implement <tt>Burgers.compute_rbf_derivative_matrix()</tt>", "d7/d6b/md_optimesh_2TASK__LIST.html#autotoc_md57", null ],
        [ "Task 1.6: Implement <tt>Burgers.compute_rbf_mass_matrix()</tt>", "d7/d6b/md_optimesh_2TASK__LIST.html#autotoc_md58", null ],
        [ "Task 1.7: Implement <tt>Burgers.compute_residual()</tt>", "d7/d6b/md_optimesh_2TASK__LIST.html#autotoc_md59", null ],
        [ "Task 1.8: Implement <tt>Burgers.residual_norm_squared()</tt>", "d7/d6b/md_optimesh_2TASK__LIST.html#autotoc_md60", null ]
      ] ],
      [ "PHASE 2: MODIFY OPTIMIZE.PY (1 hour)", "d7/d6b/md_optimesh_2TASK__LIST.html#autotoc_md62", [
        [ "Task 2.1: Add import", "d7/d6b/md_optimesh_2TASK__LIST.html#autotoc_md63", null ],
        [ "Task 2.2: Create module-level Burgers instance", "d7/d6b/md_optimesh_2TASK__LIST.html#autotoc_md64", null ],
        [ "Task 2.3: Add new objective function", "d7/d6b/md_optimesh_2TASK__LIST.html#autotoc_md65", null ],
        [ "Task 2.4: Create <tt>optimize_residual_case()</tt> function", "d7/d6b/md_optimesh_2TASK__LIST.html#autotoc_md66", null ],
        [ "Task 2.5: Create <tt>run_residual()</tt> function", "d7/d6b/md_optimesh_2TASK__LIST.html#autotoc_md67", null ]
      ] ],
      [ "PHASE 3: CREATE VALIDATION PLOTS (45 minutes)", "d7/d6b/md_optimesh_2TASK__LIST.html#autotoc_md69", [
        [ "Task 3.1: Create file <tt>optimesh/validation.py</tt>", "d7/d6b/md_optimesh_2TASK__LIST.html#autotoc_md70", null ],
        [ "Task 3.2: Implement plot_convergence()", "d7/d6b/md_optimesh_2TASK__LIST.html#autotoc_md71", null ],
        [ "Task 3.3: Implement plot_node_positions()", "d7/d6b/md_optimesh_2TASK__LIST.html#autotoc_md72", null ],
        [ "Task 3.4: Implement plot_solution_profile()", "d7/d6b/md_optimesh_2TASK__LIST.html#autotoc_md73", null ],
        [ "Task 3.5: Implement plot_residual_values()", "d7/d6b/md_optimesh_2TASK__LIST.html#autotoc_md74", null ],
        [ "Task 3.6: Implement generate_all_plots()", "d7/d6b/md_optimesh_2TASK__LIST.html#autotoc_md75", null ]
      ] ],
      [ "PHASE 4: TEST & VALIDATE (1 hour)", "d7/d6b/md_optimesh_2TASK__LIST.html#autotoc_md77", [
        [ "Task 4.1: Test imports", "d7/d6b/md_optimesh_2TASK__LIST.html#autotoc_md78", null ],
        [ "Task 4.2: Test Burgers class", "d7/d6b/md_optimesh_2TASK__LIST.html#autotoc_md79", null ],
        [ "Task 4.3: Run full optimization", "d7/d6b/md_optimesh_2TASK__LIST.html#autotoc_md80", null ],
        [ "Task 4.4: Generate validation plots", "d7/d6b/md_optimesh_2TASK__LIST.html#autotoc_md81", null ],
        [ "Task 4.5: Verify correctness", "d7/d6b/md_optimesh_2TASK__LIST.html#autotoc_md82", null ]
      ] ],
      [ "PHASE 5: DOCUMENTATION & CLEANUP (15 minutes)", "d7/d6b/md_optimesh_2TASK__LIST.html#autotoc_md84", [
        [ "Task 5.1: Add docstrings", "d7/d6b/md_optimesh_2TASK__LIST.html#autotoc_md85", null ],
        [ "Task 5.2: Code comments", "d7/d6b/md_optimesh_2TASK__LIST.html#autotoc_md86", null ],
        [ "Task 5.3: Test file cleanup", "d7/d6b/md_optimesh_2TASK__LIST.html#autotoc_md87", null ],
        [ "Task 5.4: Final verification", "d7/d6b/md_optimesh_2TASK__LIST.html#autotoc_md88", null ]
      ] ],
      [ "Summary Checklist", "d7/d6b/md_optimesh_2TASK__LIST.html#autotoc_md90", null ],
      [ "Expected Timeline", "d7/d6b/md_optimesh_2TASK__LIST.html#autotoc_md92", null ]
    ] ],
    [ "Spectral1D: High-Order Euler Solver", "d0/d30/md_README.html", [
      [ "Code Structure", "d0/d30/md_README.html#autotoc_md94", null ],
      [ "Neural Network", "d0/d30/md_README.html#autotoc_md95", null ],
      [ "Build & Compile", "d0/d30/md_README.html#autotoc_md96", null ]
    ] ],
    [ "Namespaces", "namespaces.html", [
      [ "Namespace List", "namespaces.html", "namespaces_dup" ],
      [ "Namespace Members", "namespacemembers.html", [
        [ "All", "namespacemembers.html", null ],
        [ "Functions", "namespacemembers_func.html", null ]
      ] ]
    ] ],
    [ "Classes", "annotated.html", [
      [ "Class List", "annotated.html", "annotated_dup" ],
      [ "Class Index", "classes.html", null ],
      [ "Class Hierarchy", "hierarchy.html", "hierarchy" ],
      [ "Class Members", "functions.html", [
        [ "All", "functions.html", null ],
        [ "Functions", "functions_func.html", null ]
      ] ]
    ] ],
    [ "Files", "files.html", [
      [ "File List", "files.html", "files_dup" ]
    ] ]
  ] ]
];

var NAVTREEINDEX =
[
"annotated.html",
"dd/df8/classbase_1_1RBF.html#a2873409fa647f1f9f00749142040b400"
];

var SYNCONMSG = 'click to disable panel synchronisation';
var SYNCOFFMSG = 'click to enable panel synchronisation';