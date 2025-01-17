/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cumimpactsa;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.swing.JOptionPane;

/**
 *
 * @author andy
 */
public class StatusWindow extends javax.swing.JDialog {

    private BufferedWriter bw=null;
    
    /**
     * Creates new form StatusWindow
     */
    public StatusWindow(java.awt.Frame parent, boolean modal) 
    {
        super(parent, modal);
        initComponents();
    }


    /**
     * This method is called from within the constructor to initialize the form.
     * WARNING: Do NOT modify this code. The content of this method is always
     * regenerated by the Form Editor.
     */
    @SuppressWarnings("unchecked")
    // <editor-fold defaultstate="collapsed" desc="Generated Code">//GEN-BEGIN:initComponents
    private void initComponents() {

        jButton1 = new javax.swing.JButton();
        jScrollPane1 = new javax.swing.JScrollPane();
        textArea = new javax.swing.JTextArea();
        label = new javax.swing.JLabel();

        setDefaultCloseOperation(javax.swing.WindowConstants.DISPOSE_ON_CLOSE);
        setResizable(false);

        jButton1.setText("OK");
        jButton1.setEnabled(false);
        jButton1.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jButton1ActionPerformed(evt);
            }
        });

        textArea.setEditable(false);
        textArea.setColumns(20);
        textArea.setRows(5);
        jScrollPane1.setViewportView(textArea);

        label.setText("No task");

        javax.swing.GroupLayout layout = new javax.swing.GroupLayout(getContentPane());
        getContentPane().setLayout(layout);
        layout.setHorizontalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addContainerGap()
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addComponent(label)
                    .addComponent(jScrollPane1, javax.swing.GroupLayout.PREFERRED_SIZE, 715, javax.swing.GroupLayout.PREFERRED_SIZE))
                .addContainerGap(javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
            .addGroup(javax.swing.GroupLayout.Alignment.TRAILING, layout.createSequentialGroup()
                .addContainerGap(javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                .addComponent(jButton1, javax.swing.GroupLayout.PREFERRED_SIZE, 77, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addContainerGap())
        );
        layout.setVerticalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addContainerGap(javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                .addComponent(label)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(jScrollPane1, javax.swing.GroupLayout.PREFERRED_SIZE, 290, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(jButton1))
        );

        pack();
    }// </editor-fold>//GEN-END:initComponents

    private void jButton1ActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_jButton1ActionPerformed
        this.setVisible(false);
        this.jButton1.setEnabled(false);
    }//GEN-LAST:event_jButton1ActionPerformed


    
    public void setNewText(String s)
    {
        this.textArea.setText(s);
        
        if(bw!=null) try
        {
            bw.write(s);
        }
        catch(Exception e) {println("Logging failed.");println(e);};
    }

    public synchronized void print(String s)
    {
        this.textArea.append(s);
        if(bw!=null) try
        {
            bw.write(s);
        }
        catch(Exception e) {println("Logging failed.");println(e);};
    }
    
    public synchronized void println()
    {
        this.textArea.append("\n");
        if(bw!=null) try
        {
            bw.newLine();
        }
        catch(Exception e) {println("Logging failed.");println(e);};
    }
    
    public synchronized void println(String s)
    {
        this.textArea.append("\n"+s);
        if(bw!=null) try
        {
            bw.write(s);
            bw.newLine();
        }
        catch(Exception e) {println("Logging failed.");println(e);};
    }
    
    public synchronized void setProgress(int percent)
    {
        label.setText("Progress: "+percent+"%");
    }
    
    //enables the OK button
    public void ready2bClosed()
    {
        if(bw!=null) {try {
            bw.flush();
            } catch (Exception ex) {
                println("Lost connection to log file.");
                println(ex);
            }
}
        this.jButton1.setEnabled(true);
    }

    // Variables declaration - do not modify//GEN-BEGIN:variables
    private javax.swing.JButton jButton1;
    private javax.swing.JScrollPane jScrollPane1;
    private javax.swing.JLabel label;
    private javax.swing.JTextArea textArea;
    // End of variables declaration//GEN-END:variables

    public synchronized void println(Exception e) 
    {
        textArea.append("\n\n");
        
        
        textArea.append(e.getMessage()+"\n\n");
        for(int i=0;i<Math.min(e.getStackTrace().length,10);i++)
        {
            textArea.append(e.getStackTrace()[i].toString()+"\n");
        }
        
        if(bw!=null) try
        {
            bw.newLine();
            bw.newLine();
            bw.write(e.getMessage());
            bw.newLine();
            for(int i=0;i<Math.min(e.getStackTrace().length,10);i++)
            {
                bw.write(e.getStackTrace()[i].toString());
                bw.newLine();
            }
      
        }
        catch(Exception ex)
        {
            println("Logging of exception failed");
            println(ex);
        }
        
        finally {try {bw.flush();} catch(Exception ex) {println("UNLOGGED EXCEPTION");}}
        
    }

    public synchronized void systemOutPrintln(Exception e) 
    {

        System.out.println(e.getMessage()+"\n");
        for(int i=0;i<Math.min(e.getStackTrace().length,20);i++)
        {
            System.out.println(e.getStackTrace()[i].toString()+"\n");
        }
        
        if(bw!=null) try
        {
            bw.newLine();
            bw.newLine();
            bw.write(e.getMessage());
            bw.newLine();
            for(int i=0;i<Math.min(e.getStackTrace().length,10);i++)
            {
                bw.write(e.getStackTrace()[i].toString());
                bw.newLine();
            }
            bw.flush();
        }
        catch(Exception ex)
        {
            println("Logging of exception failed");
            println(ex);
        }
        
    }
    
    void setLogFile(File logfile) 
    {
        
        FileWriter fw;
        try 
        {
            fw = new FileWriter(logfile.getAbsoluteFile());
            bw = new BufferedWriter(fw);
        } 
        catch (IOException ex) 
        {
            String message =ex.getMessage()+"\n";
            for(int i=0;i<Math.min(ex.getStackTrace().length,10);i++)
            {
                message=message+ex.getStackTrace()[i].toString()+"\n";
            }
            JOptionPane.showMessageDialog(null, message);
        }
	
    }

    public void closeLogWriter() 
    {
        try {
            bw.close();
        } catch (IOException ex) {
            JOptionPane.showMessageDialog(null, "Error closing log file.");
        }
    }

    void setProgressVisible(boolean b) {
        if(b)
        {
            this.label.setText(("No task"));
            this.label.setVisible(true);
        }
        else this.label.setVisible(false);
    }

}
