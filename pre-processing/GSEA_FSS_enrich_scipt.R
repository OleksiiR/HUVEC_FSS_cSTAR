
#if (!requireNamespace("BiocManager", quietly = TRUE))
#  install.packages("BiocManager")
#BiocManager::install("fgsea")

library("readxl")
library("dplyr")
library(fgsea)
library(ggplot2)
library(stringr)

#library("textshape")
pthr=0.05


STVs <- read.csv("STVs_SVM.csv",row.names = 1,check.names = FALSE)

for (STV in colnames(STVs)){
  gene_set = STVs[,STV]
  #names(gene_set) <- str_to_title(DEGs$Gene)
  names(gene_set) <- row.names(STVs)
  gene_set = sort(gene_set,decreasing = TRUE)
  
  # all signatures
  GO_file="msigdb_v2022.1.Hs_GMTs/msigdb.v2022.1.Hs.symbols.gmt"
  myGO = fgsea::gmtPathways(GO_file)
  # running GSEA
  res_total = fgsea(pathways=myGO, stats=gene_set) %>% as.data.frame() %>% arrange(padj)
  res_signf = as.data.frame(res_total %>% filter(padj < pthr) %>% arrange((NES)))
  # saving results
  df <- apply(res_signf,2,as.character)
  #library(xlsx)
  #write.xlsx(df,file = "genecont_GSEA.xlsx",col.names = TRUE, row.names = FALSE, append = FALSE)
  write.csv(df,paste(STV,"_STV_GSEA_all.csv",sep=""),row.names = FALSE)
  
  # C2 signature
  GO_file="msigdb_v2022.1.Hs_GMTs/c2.all.v2022.1.Hs.symbols.gmt"
  myGO = fgsea::gmtPathways(GO_file)
  # running GSEA
  res_total = fgsea(pathways=myGO, stats=gene_set) %>% as.data.frame() %>% arrange(padj)
  res_signf = as.data.frame(res_total %>% filter(padj < pthr) %>% arrange((NES)))
  # saving results
  df <- apply(res_signf,2,as.character)
  write.csv(df,paste(STV,"_STV_GSEA_C2.csv",sep=""),row.names = FALSE)
  
  # C3 all signature
  GO_file="msigdb_v2022.1.Hs_GMTs/c3.all.v2022.1.Hs.symbols.gmt"
  myGO = fgsea::gmtPathways(GO_file)
  # running GSEA
  res_total = fgsea(pathways=myGO, stats=gene_set) %>% as.data.frame() %>% arrange(padj)
  res_signf = as.data.frame(res_total %>% filter(padj < pthr) %>% arrange((NES)))
  # saving results
  df <- apply(res_signf,2,as.character)
  write.csv(df,paste(STV,"_STV_GSEA_C3.csv",sep=""),row.names = FALSE)
  
  # C3 TFs signature
  GO_file="msigdb_v2022.1.Hs_GMTs/c3.tft.gtrd.v2022.1.Hs.symbols.gmt"
  myGO = fgsea::gmtPathways(GO_file)
  # running GSEA
  res_total = fgsea(pathways=myGO, stats=gene_set) %>% as.data.frame() %>% arrange(padj)
  res_signf = as.data.frame(res_total %>% filter(padj < pthr) %>% arrange((NES)))
  # saving results
  df <- apply(res_signf,2,as.character)
  #library(xlsx)
  #write.xlsx(df,file = "genecont_GSEA.xlsx",col.names = TRUE, row.names = FALSE, append = FALSE)
  write.csv(df,paste(STV,"_STV_GSEA_C3TFs.csv",sep=""),row.names = FALSE)
  
  # C5 all signature
  GO_file="msigdb_v2022.1.Hs_GMTs/c5.all.v2022.1.Hs.symbols.gmt"
  myGO = fgsea::gmtPathways(GO_file)
  # running GSEA
  res_total = fgsea(pathways=myGO, stats=gene_set) %>% as.data.frame() %>% arrange(padj)
  res_signf = as.data.frame(res_total %>% filter(padj < pthr) %>% arrange((NES)))
  # saving results
  df <- apply(res_signf,2,as.character)
  write.csv(df,paste(STV,"_STV_GSEA_GSEA_C5.csv",sep=""),row.names = FALSE)
  
  # hallmark signature
  GO_file="msigdb_v2022.1.Hs_GMTs/h.all.v2022.1.Hs.symbols.gmt"
  myGO = fgsea::gmtPathways(GO_file)
  # running GSEA
  res_total = fgsea(pathways=myGO, stats=gene_set) %>% as.data.frame() %>% arrange(padj)
  res_signf = as.data.frame(res_total %>% filter(padj < pthr) %>% arrange((NES)))
  # saving results
  df <- apply(res_signf,2,as.character)
  write.csv(df,paste(STV,"_STV_GSEA_GSEA_HM.csv",sep=""),row.names = FALSE)
}




gene_set = STVs[,"norm_vec_remod"]
#names(gene_set) <- str_to_title(DEGs$Gene)
names(gene_set) <- row.names(STVs)
gene_set = sort(gene_set,decreasing = TRUE)

# hallmark signature
GO_file="msigdb_v2022.1.Hs_GMTs/h.all.v2022.1.Hs.symbols.gmt"
myGO = fgsea::gmtPathways(GO_file)
# running GSEA
res_total = fgsea(pathways=myGO, stats=gene_set) %>% as.data.frame() %>% arrange(padj)
res_signf = as.data.frame(res_total %>% filter(pval < pthr) %>% arrange((padj)))
# saving results
df <- apply(res_signf,2,as.character)

#plotting
#filtRes = head(res_signf, n = 5) %>% arrange(desc(NES))
filtRes = res_signf %>% arrange(desc(NES))
filtRes$Effect = ifelse(filtRes$NES > 0, "DPD up", "DPD down")
colos = setNames(c("firebrick2", "dodgerblue2"),
                 c("DPD up", "DPD down"))
g1 = ggplot(filtRes, aes(reorder(pathway, NES), NES)) +
  geom_point( aes(fill = Effect, size = size), shape=21) +
  scale_fill_manual(values = colos ) +
  scale_size_continuous(range = c(2,10)) +
  geom_hline(yintercept = 0) +
  coord_flip() +
  labs(x="Biological processes", y="Normalized Enrichment Score",
         title="STV_remod")
#pdf("TMH_TM_M3_TFs.pdf")
g1
#dev.off() 




gene_set = STVs[,"norm_vec_OSS"]
#names(gene_set) <- str_to_title(DEGs$Gene)
names(gene_set) <- row.names(STVs)
gene_set = sort(gene_set,decreasing = TRUE)

# hallmark signature
GO_file="msigdb_v2022.1.Hs_GMTs/h.all.v2022.1.Hs.symbols.gmt"
myGO = fgsea::gmtPathways(GO_file)
# running GSEA
res_total = fgsea(pathways=myGO, stats=gene_set) %>% as.data.frame() %>% arrange(padj)
res_signf = as.data.frame(res_total %>% filter(padj < pthr) %>% arrange((padj)))
# saving results
df <- apply(res_signf,2,as.character)

#plotting
filtRes = head(res_signf, n = 7) %>% arrange(desc(NES))
#filtRes = res_signf %>% arrange(desc(NES))
filtRes$Effect = ifelse(filtRes$NES > 0, "DPD up", "DPD down")
colos = setNames(c("firebrick2", "dodgerblue2"),
                 c("DPD up", "DPD down"))
g1 = ggplot(filtRes, aes(reorder(pathway, NES), NES)) +
  geom_point( aes(fill = Effect, size = size), shape=21) +
  scale_fill_manual(values = colos ) +
  scale_size_continuous(range = c(2,10)) +
  geom_hline(yintercept = 0) +
  coord_flip() +
  labs(x="Biological processes", y="Normalized Enrichment Score",
       title="STV_OSS")
#pdf("TMH_TM_M3_TFs.pdf")
g1
#dev.off() 


